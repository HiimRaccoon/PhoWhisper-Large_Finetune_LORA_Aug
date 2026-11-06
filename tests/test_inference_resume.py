import json
from pathlib import Path
import pytest


def journal_fixture(tmp_path):
    from test_outputs import output_fixture
    from asr.paths import write_jsonl_atomic
    rows,predictions,exported=output_fixture(tmp_path)
    manifest=tmp_path/'manifest.jsonl'; write_jsonl_atomic(manifest,[r.as_dict() for r in rows])
    return rows,predictions,exported,manifest,tmp_path/'transcripts'


def test_interrupted_journal_resumes_only_uncommitted_utterances(tmp_path):
    from asr.inference import transcribe_ordered
    rows,predictions,exported,manifest,output=journal_fixture(tmp_path)
    seen=[]
    def interrupt(row):
        seen.append(row.utterance_id)
        if len(seen)==2: raise RuntimeError('worker interrupted')
        return predictions[0]
    with pytest.raises(RuntimeError,match='interrupted'):
        transcribe_ordered(manifest,exported,output,interrupt,{'max_new_tokens':440})
    assert not (output/'output-validation.json').exists()
    journal=output/'.inference-progress/results.partial.jsonl'
    assert len(journal.read_bytes().splitlines())==1
    with journal.open('ab') as stream: stream.write(b'{"torn')
    resumed=[]
    def predict(row):
        resumed.append(row.utterance_id)
        return predictions[rows.index(row)]
    report=transcribe_ordered(manifest,exported,output,predict,{'max_new_tokens':440})
    assert report.valid and resumed==['s1/history_002','s2/current']
    assert (output/'.inference-progress/recovery.json').is_file()
    assert transcribe_ordered(manifest,exported,output,lambda _:pytest.fail('already complete'),{'max_new_tokens':440}).valid


@pytest.mark.parametrize('corruption',['complete_invalid_json','duplicate','wrong_audio','wrong_model','generation_config','manifest'])
def test_resume_refuses_committed_corruption_or_identity_change(tmp_path,corruption):
    from asr.inference import transcribe_ordered
    from asr.paths import write_jsonl_atomic
    rows,predictions,exported,manifest,output=journal_fixture(tmp_path)
    def stop(row):
        if row!=rows[0]: raise RuntimeError('stop')
        return predictions[0]
    with pytest.raises(RuntimeError): transcribe_ordered(manifest,exported,output,stop,{'setting':1})
    journal=output/'.inference-progress/results.partial.jsonl'
    generation={'setting':1}
    if corruption=='complete_invalid_json': journal.write_bytes(b'{invalid}\n')
    if corruption=='duplicate':
        with journal.open('ab') as stream: stream.write(journal.read_bytes())
    if corruption=='wrong_audio':
        predictions[0]['wav_sha256']='wrong'; write_jsonl_atomic(journal,[predictions[0]])
    if corruption=='wrong_model': exported.files['model.safetensors']='changed'
    if corruption=='generation_config': generation={'setting':2}
    if corruption=='manifest':
        rows[0]=__import__('dataclasses').replace(rows[0],audio='audio/different.wav')
        write_jsonl_atomic(manifest,[r.as_dict() for r in rows])
    with pytest.raises(ValueError):
        transcribe_ordered(manifest,exported,output,lambda _:pytest.fail('must reject before generation'),generation)


def test_failed_generation_is_diagnostic_and_not_committed(tmp_path):
    from asr.inference import transcribe_ordered,GenerationFailure
    rows,predictions,exported,manifest,output=journal_fixture(tmp_path)
    def fail(row): raise GenerationFailure({'utterance_id':row.utterance_id,'reason':'limit','tokens':[6]*440})
    with pytest.raises(GenerationFailure): transcribe_ordered(manifest,exported,output,fail,{})
    assert (output/'.inference-progress/results.partial.jsonl').read_bytes()==b''
    assert (output/'.inference-progress/last-error.json').is_file()


def test_publication_can_resume_between_files_without_overwriting_unknown_data(tmp_path,monkeypatch):
    import os
    from asr.inference import transcribe_ordered
    rows,predictions,exported,manifest,output=journal_fixture(tmp_path)
    original=os.link
    def crash(src,dst,*args,**kwargs):
        if Path(dst).name=='train_transcripts.jsonl': raise OSError('publication interruption')
        return original(src,dst,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(os,'link',crash)
        with pytest.raises(OSError,match='interruption'):
            transcribe_ordered(manifest,exported,output,lambda r:predictions[rows.index(r)],{})
    assert (output/'utterances.jsonl').exists() and not (output/'output-validation.json').exists()
    assert transcribe_ordered(manifest,exported,output,lambda _:pytest.fail('journal already complete'),{}).valid
    (output/'utterances.jsonl').write_text('unknown',encoding='utf-8')
    with pytest.raises(ValueError): transcribe_ordered(manifest,exported,output,lambda _:None,{})


def test_writer_lock_prevents_parallel_journals(tmp_path):
    from filelock import FileLock
    from asr.inference import transcribe_ordered
    rows,predictions,exported,manifest,output=journal_fixture(tmp_path)
    private=output/'.inference-progress'; private.mkdir(parents=True)
    with FileLock(str(private/'.writer.lock')):
        with pytest.raises(RuntimeError,match='writer'):
            transcribe_ordered(manifest,exported,output,lambda _:None,{})


def test_fsync_interruption_recovers_durable_line_once(tmp_path,monkeypatch):
    import os
    from asr.inference import transcribe_ordered
    rows,predictions,exported,manifest,output=journal_fixture(tmp_path)
    real=os.fsync; stopped=False
    def fail_once(fd):
        nonlocal stopped
        real(fd)
        journal=output/'.inference-progress/results.partial.jsonl'
        if journal.exists() and journal.stat().st_size and not stopped:
            stopped=True
            raise OSError('interrupted after durable write')
    with monkeypatch.context() as patch:
        patch.setattr(os,'fsync',fail_once)
        with pytest.raises(OSError,match='durable'):
            transcribe_ordered(manifest,exported,output,lambda r:predictions[rows.index(r)],{})
    resumed=[]
    def predict(row):
        resumed.append(row.utterance_id); return predictions[rows.index(row)]
    assert transcribe_ordered(manifest,exported,output,predict,{}).valid
    assert resumed==['s1/history_002','s2/current']
