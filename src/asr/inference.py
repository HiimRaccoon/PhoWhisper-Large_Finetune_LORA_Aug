"""Offline greedy ASR with an identity-bound, durable ordered prediction journal."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import torch
from filelock import FileLock,Timeout

from .config import canonical_hash
from .contracts import UtteranceRow,OutputValidation,RuntimeProfile
from .export import locked_generation
from .outputs import model_identity,group_transcripts,validate_outputs,validate_prediction
from .paths import read_json,read_jsonl,sha256_file,write_json_atomic,write_jsonl_atomic
from .processor import prepare_features


class GenerationFailure(RuntimeError):
    def __init__(self,evidence):
        self.evidence=evidence
        super().__init__(f"generation failed: {evidence['utterance_id']}: {evidence['reason']}")


def predict_utterance(model,processor,row,audio,exported):
    model.eval()
    parameter=next(model.parameters())
    features=prepare_features(audio,processor.feature_extractor)
    inputs={'input_features':features['input_features'].to(parameter.device,dtype=parameter.dtype),
            'attention_mask':features['attention_mask'].to(parameter.device)}
    generation=locked_generation(model,processor)
    # Whisper's default Tensor return strips EOS. ModelOutput retains the decoder prefix and EOS.
    with torch.inference_mode():
        output=model.generate(**inputs,generation_config=generation,temperature=0.0,
                              return_dict_in_generate=True)
    sequence=output.sequences[0].detach().cpu().tolist()
    vocab=processor.tokenizer.get_vocab()
    prefix=[vocab[t] for t in ['<|startoftranscript|>','<|vi|>','<|transcribe|>','<|notimestamps|>']]
    if sequence[:4]!=prefix:
        raise GenerationFailure({'utterance_id':row.utterance_id,'reason':'decoder_prefix','tokens':sequence})
    tokens=sequence[4:]
    eos=generation.eos_token_id
    ended=eos in tokens
    if ended: tokens=tokens[:tokens.index(eos)+1]
    text=processor.batch_decode([tokens],skip_special_tokens=True)[0].strip()
    evidence={'utterance_id':row.utterance_id,'generated_tokens':len(tokens),'eos':ended,
              'hit_limit':len(tokens)>=440,'tokens':tokens}
    if not ended or not text or len(tokens)>440:
        raise GenerationFailure({**evidence,'reason':'limit' if not ended and len(tokens)>=440 else 'unfinished' if not ended else 'empty' if not text else 'limit'})
    return {'sample_id':row.sample_id,'turn_id':row.turn_id,'utterance_id':row.utterance_id,
            'audio':row.audio,'wav_sha256':row.wav_sha256,'transcript':text,
            'model_run_id':exported.metadata['provenance']['run_id'],'model_identity':model_identity(exported),
            'generation':{k:v for k,v in {**evidence,'status':'complete'}.items() if k!='utterance_id'}}


def _install(candidate,destination):
    """Atomic no-overwrite publication; identical crash leftovers are safe to reuse."""
    if destination.exists():
        if sha256_file(destination)!=sha256_file(candidate):
            raise ValueError(f'unknown/different published artifact: {destination.name}')
    else:
        os.link(candidate,destination)


def transcribe_ordered(manifest,exported,output_dir,predict,resolved_generation):
    """Journal component shared by official inference and small crash-recovery fixtures."""
    output=Path(output_dir); private=output/'.inference-progress'
    private.mkdir(parents=True,exist_ok=True)
    try:
        with FileLock(str(private/'.writer.lock'),timeout=0):
            rows=[UtteranceRow(**r) for r in read_jsonl(manifest)]
            if len({r.utterance_id for r in rows})!=len(rows): raise ValueError('duplicate manifest ID')
            identity={'schema_version':1,'model_identity':model_identity(exported),
                      'run_id':exported.metadata['provenance']['run_id'],'manifest_hash':sha256_file(manifest),
                      'generation':resolved_generation}
            identity_file=private/'identity.json'
            if identity_file.exists():
                if read_json(identity_file)!=identity: raise ValueError('inference continuation identity mismatch')
            else:
                if any(p.name!='.writer.lock' for p in private.iterdir()):
                    raise ValueError('journal exists without identity')
                write_json_atomic(identity_file,identity)
            utterances=output/'utterances.jsonl'; sources=output/'train_transcripts.jsonl'
            completion=output/'output-validation.json'
            if completion.exists():
                report=validate_outputs(manifest,utterances,sources,exported)
                if not report.valid or read_json(completion)!=report.as_dict():
                    raise ValueError('completed output hash/schema/identity mismatch')
                return report
            journal=private/'results.partial.jsonl'
            if not journal.exists(): journal.touch(exist_ok=False)
            payload=journal.read_bytes()
            if payload and not payload.endswith(b'\n'):
                end=payload.rfind(b'\n')+1
                write_json_atomic(private/'recovery.json',{'discarded_unterminated_bytes':len(payload)-end,
                                  'discarded_sha256':__import__('hashlib').sha256(payload[end:]).hexdigest()})
                with journal.open('r+b') as stream:
                    stream.truncate(end); stream.flush(); os.fsync(stream.fileno())
                payload=payload[:end]
            try:
                predictions=[json.loads(line) for line in payload.splitlines()]
            except (ValueError,UnicodeError) as exc:
                raise ValueError('corrupt committed journal record') from exc
            if len(predictions)>len(rows): raise ValueError('journal exceeds manifest coverage')
            for row,prediction in zip(rows,predictions): validate_prediction(row,prediction,exported)
            with journal.open('ab') as stream:
                for row in rows[len(predictions):]:
                    try:
                        prediction=predict(row)
                        validate_prediction(row,prediction,exported)
                        line=(json.dumps(prediction,ensure_ascii=False,allow_nan=False,sort_keys=True)+'\n').encode('utf-8')
                        stream.write(line); stream.flush(); os.fsync(stream.fileno())
                        predictions.append(prediction)
                    except Exception as exc:
                        write_json_atomic(private/'last-error.json',getattr(exc,'evidence',
                                          {'utterance_id':row.utterance_id,'error':f'{type(exc).__name__}: {exc}'}))
                        raise
            candidate_u=private/'candidate-utterances.jsonl'; candidate_s=private/'candidate-sources.jsonl'
            write_jsonl_atomic(candidate_u,predictions)
            write_jsonl_atomic(candidate_s,group_transcripts(rows,predictions))
            report=validate_outputs(manifest,candidate_u,candidate_s,exported)
            if not report.valid: raise ValueError(f'output validation failed: {report.errors}')
            write_json_atomic(private/'candidate-hashes.json',report.files)
            _install(candidate_u,utterances); _install(candidate_s,sources)
            candidate_report=private/'candidate-validation.json'
            write_json_atomic(candidate_report,report.as_dict())
            _install(candidate_report,completion)
            return report
    except Timeout as exc:
        raise RuntimeError('another inference writer owns the journal') from exc


def transcribe_train(config,manifest,exported,output_dir) -> OutputValidation:
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    import transformers
    from .export import load_export
    from .paths import project_path,audio_path
    from .data import read_pcm16
    from .precision import require_gpu_allocation
    from .preflight import load_audited_rows,current_runtime_identity
    from .artifacts import set_status
    require_gpu_allocation(os.environ,socket.gethostname())
    exported=load_export(project_path(config,exported.path))
    manifest=project_path(config,manifest); output=project_path(config,output_dir)
    rows,audit=load_audited_rows(config,verify_audio=True)
    if (sha256_file(manifest)!=exported.identity.manifest_hash or audit['manifest_sha256']!=exported.identity.manifest_hash
            or config.method_hash!=exported.identity.method_hash):
        raise ValueError('inference dataset/config identity mismatch')
    run=exported.path.parent
    record=read_json(run/'run-manifest.json')
    if (record['identity']!=exported.identity.as_dict() or record['run_id']!=exported.metadata['provenance']['run_id']
            or record['status'] not in {'export_complete','transcripts_complete'} or output!=run/'transcripts'):
        raise ValueError('inference run/output identity mismatch')
    if current_runtime_identity(RuntimeProfile(**exported.metadata['provenance']['runtime']))!=exported.identity.runtime_hash:
        raise ValueError('inference GPU/driver identity changed since verified export')
    processor=WhisperProcessor.from_pretrained(str(exported.path),local_files_only=True,trust_remote_code=False)
    model=WhisperForConditionalGeneration.from_pretrained(str(exported.path),local_files_only=True,
                      trust_remote_code=False,torch_dtype=torch.float16).to('cuda').eval()
    generation=locked_generation(model,processor)
    resolved={'method':config.inference.as_dict() if hasattr(config.inference,'as_dict') else config.method_dict()['inference'],
              'generation':generation.to_dict(),'temperature':0.0,'transformers':transformers.__version__}
    def predict(row):
        audio,metadata=read_pcm16(audio_path(config.paths.train_root,row.audio))
        if metadata['wav_sha256']!=row.wav_sha256: raise ValueError(f'audio changed: {row.utterance_id}')
        return predict_utterance(model,processor,row,audio,exported)
    report=transcribe_ordered(manifest,exported,output,predict,resolved)
    if report.valid and record['status']!='transcripts_complete':
        set_status(run,'transcripts_complete',outputs=report.as_dict())
        summary=read_json(run/'run-summary.json')
        summary.update(status='transcripts_complete',outputs=report.as_dict(),export_manifest_sha256=sha256_file(exported.path/'export-manifest.json'))
        write_json_atomic(run/'run-summary.json',summary)
    return report
