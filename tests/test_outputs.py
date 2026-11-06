from dataclasses import replace
from pathlib import Path
import pytest


def output_fixture(tmp_path):
    from asr.contracts import UtteranceRow,ExportResult
    from test_checkpoint import identity_fixture
    from asr.config import canonical_hash
    rows=[UtteranceRow('s1','current','s1/current','audio/s1.wav','SECRET',wav_sha256='a'*64),
          UtteranceRow('s1','history_002','s1/history_002','audio/h.wav','HIDDEN',history_index=2,wav_sha256='b'*64),
          UtteranceRow('s2','current','s2/current','audio/s2.wav','SECRET',wav_sha256='c'*64)]
    exported=ExportResult(tmp_path/'merged',True,identity_fixture(),{'model.safetensors':'d'*64},
                          {'provenance':{'run_id':'official'},'generation':{'max_new_tokens':440}})
    predictions=[{'sample_id':r.sample_id,'turn_id':r.turn_id,'utterance_id':r.utterance_id,'audio':r.audio,
          'wav_sha256':r.wav_sha256,'transcript':f'prediction {i}','model_run_id':'official',
          'model_identity':canonical_hash({'identity':exported.identity.as_dict(),'files':exported.files}),
          'generation':{'status':'complete','eos':True,'generated_tokens':10,'hit_limit':False}}
          for i,r in enumerate(rows)]
    return rows,predictions,exported


def test_grouping_preserves_original_history_index_and_empty_history(tmp_path):
    from asr.outputs import group_transcripts
    rows,predictions,_=output_fixture(tmp_path)
    result=group_transcripts(rows,predictions)
    assert [r['sample_id'] for r in result]==['s1','s2']
    assert result[0]['history_user_turns'][0]['history_index']==2
    assert result[0]['current']['transcript']=='prediction 0'
    assert result[1]['history_user_turns']==[]
    assert 'SECRET' not in str(result) and 'HIDDEN' not in str(result)


@pytest.mark.parametrize('corruption',['missing','extra','duplicate','identity','empty','unfinished','history'])
def test_validator_rejects_incomplete_or_mismatched_outputs(tmp_path,corruption):
    from asr.outputs import group_transcripts,validate_outputs
    from asr.paths import write_jsonl_atomic
    rows,predictions,exported=output_fixture(tmp_path)
    source=group_transcripts(rows,predictions)
    if corruption=='missing': predictions.pop()
    if corruption=='extra': predictions.append(dict(predictions[-1],utterance_id='extra'))
    if corruption=='duplicate': predictions[-1]=predictions[0]
    if corruption=='identity': predictions[0]['model_identity']='wrong'
    if corruption=='empty': predictions[0]['transcript']=' '
    if corruption=='unfinished': predictions[0]['generation']['eos']=False
    if corruption=='history': source[0]['history_user_turns'][0]['history_index']=0
    manifest=tmp_path/'manifest.jsonl'; utterances=tmp_path/'utterances.jsonl'; sources=tmp_path/'sources.jsonl'
    write_jsonl_atomic(manifest,[r.as_dict() for r in rows]); write_jsonl_atomic(utterances,predictions); write_jsonl_atomic(sources,source)
    result=validate_outputs(manifest,utterances,sources,exported)
    assert not result.valid and result.errors


def test_validator_reports_counts_and_file_hashes(tmp_path):
    from asr.outputs import group_transcripts,validate_outputs
    from asr.paths import write_jsonl_atomic,sha256_file
    rows,predictions,exported=output_fixture(tmp_path)
    paths=[tmp_path/name for name in ['manifest.jsonl','utterances.jsonl','sources.jsonl']]
    for path,records in zip(paths,[[r.as_dict() for r in rows],predictions,group_transcripts(rows,predictions)]):
        write_jsonl_atomic(path,records)
    result=validate_outputs(*paths,exported)
    assert result.valid and result.utterance_count==3 and result.source_count==2
    assert result.files['utterances.jsonl']==sha256_file(paths[1])
