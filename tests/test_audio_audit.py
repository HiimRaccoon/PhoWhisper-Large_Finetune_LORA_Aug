import json
import pytest


def test_pcm_integrity_rejects_truncation_and_wrong_format(tmp_path, make_wav):
    from asr.data import read_pcm16
    path=make_wav(tmp_path/'good.wav')
    audio, meta=read_pcm16(path)
    assert audio.tolist() == pytest.approx([100/32768,200/32768,-300/32768,400/32768])
    assert meta['duration_seconds'] == 0.00025
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError,match='truncated'):
        read_pcm16(path)
    with pytest.raises(ValueError,match='format'):
        read_pcm16(make_wav(tmp_path/'wrong.wav',rate=8000))


def test_audit_preserves_duplicates_and_blocks_transcript_conflicts(local_config, make_wav):
    from asr.audit import audit_train
    root=local_config.paths.train_root
    make_wav(root/'audio/one.wav')
    rows=[{'id':'s1','audio':'audio/one.wav','audio_text':'Café','history':[]},
          {'id':'s2','audio':'audio/one.wav','audio_text':'  Cafe\u0301 ','history':[]}]
    path=root/'train.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    result=audit_train(local_config,local_config.paths.manifest_dir)
    assert result.utterance_count == 2
    assert result.duplicate_groups == 1
    assert result.conflict_groups == 0
    assert len(result.manifest_path.read_text(encoding='utf-8').splitlines()) == 2
    assert result.valid is False  # miniature fixture must never pass official hash/count gates
    rows[1]['audio_text']='Different reference'
    path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    result=audit_train(local_config,local_config.paths.manifest_dir)
    assert result.conflict_groups == 1
    assert any(e['kind']=='transcript_conflict' for e in result.errors)


def test_missing_audio_is_reported_and_output_cannot_escape_project(local_config):
    from asr.audit import audit_train
    root=local_config.paths.train_root; root.mkdir(parents=True)
    (root/'train.jsonl').write_text(json.dumps({'id':'missing','audio':'audio/no.wav','audio_text':'x','history':[]}),encoding='utf-8')
    result=audit_train(local_config,local_config.paths.manifest_dir)
    assert result.valid is False
    assert any(e.get('utterance_id')=='missing/current' for e in result.errors)
    with pytest.raises(ValueError):
        audit_train(local_config,local_config.paths.project_root.parent/'outside-output')
