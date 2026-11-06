"""Terminal-only FP32 merge, FP16 publication and independent offline verification."""
from __future__ import annotations

import copy
import gc
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid

import torch
from filelock import FileLock

from .config import canonical_hash
from .contracts import ArtifactIdentity,ExportResult,RuntimeProfile
from .paths import read_json,resolve_beneath,sha256_file,write_json_atomic


def verify_terminal_checkpoint(state,budget,expected) -> None:
    p=state.progress
    valid=(state.identity==expected and p.completed_epochs==10
           and p.global_update==budget.total_updates
           and p.global_micro_step==10*budget.utterances_per_epoch
           and p.pending_micro_steps==0
           and p.successful_updates_by_epoch==[budget.updates_per_epoch]*10
           and p.utterances_completed_by_epoch==[budget.utterances_per_epoch]*10
           and (p.epoch,p.next_position) in [(10,0),(9,budget.utterances_per_epoch)]
           and state.metadata.get('kind')=='terminal'
           and len(state.metadata.get('adapter_sha256',''))==64
           and state.metadata.get('pending_gradient_count')==0
           and p.permutation_hash==canonical_hash(state.permutation)
           and sorted(state.permutation)==list(range(budget.utterances_per_epoch))
           and state.scheduler.get('last_epoch')==budget.total_updates
           and all(float(v['step'])==budget.total_updates for v in state.optimizer['state'].values())
           and bool(state.optimizer['state'])
           and all(g['lr']==0 for g in state.optimizer['param_groups']))
    if not valid:
        raise ValueError('checkpoint does not prove complete terminal training')


def locked_generation(model,processor):
    """Fully resolve settings instead of inheriting sampling or timestamp defaults."""
    generation=copy.deepcopy(model.generation_config)
    vocab=processor.tokenizer.get_vocab()
    generation.lang_to_id={t:i for t,i in vocab.items() if len(t)==6 and t.startswith('<|') and t.endswith('|>')}
    generation.task_to_id={task:vocab[f'<|{task}|>'] for task in ['transcribe','translate']}
    generation.no_timestamps_token_id=vocab['<|notimestamps|>']
    for name,value in dict(language='vi',task='transcribe',do_sample=False,num_beams=1,
                           max_new_tokens=440,return_timestamps=False,temperature=1.0,
                           forced_decoder_ids=None,use_cache=True,
                           compression_ratio_threshold=None,logprob_threshold=None,
                           no_speech_threshold=None,return_token_timestamps=False).items():
        setattr(generation,name,value)
    generation.is_multilingual=True
    return generation


def _inventory(path):
    return {p.relative_to(path).as_posix():sha256_file(p) for p in sorted(path.rglob('*'))
            if p.is_file() and p.name!='export-manifest.json'}


def load_export(path: Path) -> ExportResult:
    path=Path(path).resolve()
    record=read_json(path/'export-manifest.json')
    if record.get('schema_version')!=1 or record.get('valid') is not True:
        raise ValueError('invalid export completion manifest')
    for name in record['files']:
        resolve_beneath(path,name)
    if _inventory(path)!=record['files']:
        raise ValueError('export file inventory/hash mismatch')
    return ExportResult(path,True,ArtifactIdentity(**record['identity']),record['files'],record['metadata'])


def verify_local_export(path,device='cpu'):
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    torch.set_num_threads(1)
    model=WhisperForConditionalGeneration.from_pretrained(str(path),local_files_only=True,
               trust_remote_code=False,torch_dtype=torch.float16).to(device).eval()
    processor=WhisperProcessor.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
    if hasattr(model,'peft_config') or any('lora_' in n for n,_ in model.named_parameters()):
        raise ValueError('export still requires an adapter')
    if any(p.dtype!=torch.float16 or not torch.isfinite(p).all() for p in model.parameters() if p.is_floating_point()):
        raise ValueError('export has non-finite or non-FP16 weights')
    with torch.inference_mode():
        result=model.generate(torch.zeros(1,80,3000,dtype=torch.float16,device=device),
               attention_mask=torch.ones(1,3000,dtype=torch.long,device=device),
               generation_config=locked_generation(model,processor),max_new_tokens=1,temperature=0.0)
    return {'valid':True,'dtype':'float16','technical_probe_max_new_tokens':1,
            'generated_tokens':int(result.numel()),'local_files_only':True,'plain_whisper_model':True}


def merge_and_export(model,processor,output_dir,identity,provenance,device='cuda') -> ExportResult:
    output=Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output.parent/'.export.lock'),timeout=0):
        if output.exists(): raise FileExistsError(output)
        if any(p.dtype!=torch.float32 for p in model.parameters() if p.is_floating_point()):
            raise ValueError('merge inputs must retain FP32 storage')
        temporary=output.parent/f'.tmp-export-{uuid.uuid4().hex}'
        temporary.mkdir()
        model.eval()
        # The reference is independently rounded to FP16. Merge the untouched FP32 source.
        reference=copy.deepcopy(model).half().to(device).eval()
        features=torch.zeros(1,80,3000,dtype=torch.float16,device=device)
        mask=torch.ones(1,3000,dtype=torch.long,device=device)
        ids=torch.tensor([[model.config.decoder_start_token_id]],device=device)
        with torch.inference_mode():
            before=reference(input_features=features,attention_mask=mask,decoder_input_ids=ids).logits.float().cpu()
        del reference
        gc.collect()
        if device=='cuda': torch.cuda.empty_cache()
        merged=model.merge_and_unload(safe_merge=True).half().to(device).eval()
        merged.config.use_cache=True
        merged.generation_config=locked_generation(merged,processor)
        with torch.inference_mode():
            after=merged(input_features=features,attention_mask=mask,decoder_input_ids=ids).logits.float().cpu()
        if not torch.isfinite(after).all(): raise ValueError('non-finite merged logits')
        torch.testing.assert_close(after,before,atol=0.05,rtol=0.05)
        comparison={'passed':True,'atol':0.05,'rtol':0.05,'max_abs_error':float((after-before).abs().max())}
        if any(p.dtype!=torch.float16 or not torch.isfinite(p).all() for p in merged.parameters() if p.is_floating_point()):
            raise ValueError('merged floating weights must be finite FP16')
        merged.save_pretrained(str(temporary),safe_serialization=True)
        processor.save_pretrained(str(temporary))
        write_json_atomic(temporary/'export-provenance.json',provenance)
        write_json_atomic(temporary/'comparison.json',comparison)
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                 PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        child=subprocess.run([sys.executable,'-m','asr.export','--verify',str(temporary),'--device',device],
                             env=env,capture_output=True,text=True,check=False)
        if child.returncode:
            raise RuntimeError(f'fresh-process offline reload failed: {child.stderr[-4000:]}')
        reload=read_json(temporary/'reload.json')
        metadata={'comparison':comparison,'fresh_process_reload':reload,'dtype':'float16',
                  'provenance':provenance,'generation':merged.generation_config.to_dict()}
        result=ExportResult(output,True,identity,_inventory(temporary),metadata)
        write_json_atomic(temporary/'export-manifest.json',result.as_dict())
        os.rename(temporary,output)
        return load_export(output)


def export_final(config,checkpoint: Path,output_dir: Path) -> ExportResult:
    from .checkpoint import load_checkpoint,restore_adapter
    from .paths import project_path
    from .preflight import current_input_identity
    from .precision import require_gpu_allocation
    from .model import build_training_model
    from .processor import load_processor
    from .training import compute_budget
    from .artifacts import set_status
    checkpoint=project_path(config,checkpoint); output=project_path(config,output_dir)
    if output.exists(): raise FileExistsError(output)
    metadata=read_json(checkpoint/'metadata.json')
    identity=ArtifactIdentity(**metadata['identity'])
    state=load_checkpoint(checkpoint,identity)
    verify_terminal_checkpoint(state,compute_budget(config.data.expected_utterances),identity)
    run=checkpoint.parent.parent.parent
    run_record=read_json(run/'run-manifest.json')
    if (run_record.get('identity')!=identity.as_dict() or metadata.get('run_id')!=run.name
            or run_record.get('mode')=='smoke'):
        raise ValueError('final checkpoint/run identity mismatch')
    if output != run/'merged-fp16':
        raise ValueError('official export must be runs/<run_id>/merged-fp16')
    require_gpu_allocation(os.environ,socket.gethostname())
    runtime=RuntimeProfile(**metadata['runtime'])
    current,rows,snapshot,_,_=current_input_identity(config,runtime)
    if current!=identity: raise ValueError('merge input identity changed')
    bundle=build_training_model(config,snapshot,runtime)
    if state.metadata['target_names']!=bundle.target_names:
        raise ValueError('final adapter scope mismatch')
    restore_adapter(bundle.model,state.adapter)
    provenance={'run_id':run.name,'checkpoint':str(checkpoint),'adapter_sha256':metadata['adapter_sha256'],
                'base_revision':snapshot.revision,'target_names':bundle.target_names,
                'runtime':runtime.as_dict(),'terminal_progress':state.progress.as_dict()}
    result=merge_and_export(bundle.model,load_processor(snapshot),output,identity,provenance)
    set_status(run,'export_complete',export=str(output),export_manifest_sha256=sha256_file(output/'export-manifest.json'))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--verify',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu'); args=parser.parse_args()
    write_json_atomic(args.verify/'reload.json',verify_local_export(args.verify,args.device))
