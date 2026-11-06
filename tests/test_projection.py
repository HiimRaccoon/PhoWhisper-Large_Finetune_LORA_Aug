import json
import pytest


def test_projection_keeps_original_history_indices_and_own_references(tmp_path):
    from asr.data import project_sources
    source = {"id":"s1", "audio":"audio/s1/current.wav", "audio_text":"Current", "label":{"secret":1},
              "history":[{"role":"user","audio":"audio/s1/u1.wav","audio_text":"First"},
                         {"role":"assistant","text":"Reply"},
                         {"role":"user","audio":"audio/s1/u2.wav","audio_text":"Second"}]}
    path=tmp_path / 'train.jsonl'
    path.write_text(json.dumps(source),encoding='utf-8')
    rows=project_sources(path)
    assert [(r.turn_id,r.reference,r.history_index) for r in rows] == [
        ('current','Current',None),('history_000','First',0),('history_002','Second',2)]
    assert [r.utterance_id for r in rows] == ['s1/current','s1/history_000','s1/history_002']
    assert 'secret' not in repr(rows)


def test_duplicate_source_ids_and_missing_user_audio_are_rejected(tmp_path):
    from asr.data import project_sources
    p=tmp_path/'train.jsonl'
    row={"id":"same","audio":"audio/x.wav","audio_text":"x","history":[]}
    p.write_text((json.dumps(row)+'\n')*2,encoding='utf-8')
    with pytest.raises(ValueError,match='duplicate'):
        project_sources(p)
    row['history']=[{'role':'user','audio_text':'missing'}]
    p.write_text(json.dumps(row),encoding='utf-8')
    with pytest.raises(ValueError,match='history_000'):
        project_sources(p)
