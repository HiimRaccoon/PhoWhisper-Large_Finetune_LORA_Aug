"""Strict original-order transcript mapping and complete artifact validation."""
from .config import canonical_hash
from .contracts import OutputValidation,UtteranceRow
from .paths import read_jsonl,sha256_file


def model_identity(exported):
    return canonical_hash({'identity':exported.identity.as_dict(),'files':exported.files})


def validate_prediction(row,prediction,exported):
    for key in ['sample_id','turn_id','utterance_id','audio','wav_sha256']:
        if prediction.get(key)!=getattr(row,key):
            raise ValueError(f'{row.utterance_id}: prediction {key} mismatch')
    if (prediction.get('model_run_id')!=exported.metadata['provenance']['run_id']
            or prediction.get('model_identity')!=model_identity(exported)):
        raise ValueError(f'{row.utterance_id}: model identity mismatch')
    if not isinstance(prediction.get('transcript'),str) or not prediction['transcript'].strip():
        raise ValueError(f'{row.utterance_id}: empty transcript')
    generation=prediction.get('generation',{})
    count=generation.get('generated_tokens')
    if (generation.get('status')!='complete' or generation.get('eos') is not True
            or type(count) is not int or not 1<=count<=440):
        raise ValueError(f'{row.utterance_id}: incomplete generation')


def group_transcripts(rows,predictions):
    if len(rows)!=len(predictions): raise ValueError('prediction coverage mismatch')
    grouped={}
    for row,prediction in zip(rows,predictions):
        if prediction.get('utterance_id')!=row.utterance_id:
            raise ValueError('prediction order mismatch')
        sample=grouped.setdefault(row.sample_id,{'schema_version':1,'sample_id':row.sample_id,
                    'model_run_id':prediction['model_run_id'],'current':None,'history_user_turns':[]})
        item={k:prediction[k] for k in ['turn_id','utterance_id','audio','transcript']}
        if row.turn_id=='current':
            if sample['current'] is not None: raise ValueError('duplicate current')
            sample['current']=item
        else:
            if row.history_index is None or row.turn_id!=f'history_{row.history_index:03d}':
                raise ValueError('invalid original history index')
            sample['history_user_turns'].append({'history_index':row.history_index,**item})
    if any(sample['current'] is None for sample in grouped.values()): raise ValueError('missing current')
    return list(grouped.values())


def validate_outputs(manifest,utterance_file,source_file,exported) -> OutputValidation:
    errors=[]; predictions=[]; sources=[]; files={}
    try:
        rows=[UtteranceRow(**r) for r in read_jsonl(manifest)]
        predictions=read_jsonl(utterance_file); sources=read_jsonl(source_file)
        if len({r.utterance_id for r in rows})!=len(rows): raise ValueError('duplicate manifest ID')
        if len(predictions)!=len(rows): raise ValueError('utterance coverage mismatch')
        for row,prediction in zip(rows,predictions): validate_prediction(row,prediction,exported)
        if group_transcripts(rows,predictions)!=sources: raise ValueError('source mapping/content mismatch')
        files={'utterances.jsonl':sha256_file(utterance_file),'train_transcripts.jsonl':sha256_file(source_file)}
    except (ValueError,TypeError,KeyError,OSError) as exc:
        errors.append(str(exc))
    return OutputValidation(not errors,len(predictions),len(sources),exported.identity,files,errors)
