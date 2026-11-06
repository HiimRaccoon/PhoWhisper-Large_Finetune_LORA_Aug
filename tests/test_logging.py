import json


def test_log_records_are_parseable_and_keep_attempt_and_fallback_context(tmp_path):
    from asr.logging import RunLogger
    path=tmp_path/'events.jsonl'
    with RunLogger(path,'run1','attempt1') as logger:
        logger.log('optimizer_update',epoch=0,position=16,micro_step=16,global_update=1,
                   loss=2.0,scaled_loss=0.125,window_size=16,lr=0.0,gradient_norm=0.5,
                   dtype='bfloat16',scaler={'enabled':False},gpu_memory={'allocated':100},
                   elapsed_seconds=1.0,throughput=16.0)
        logger.log('augmentation',utterance_id='s/current',epoch=0,branch='augmentation',fallbacks=['invalid_peak'])
    records=[json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]['lr']==0 and records[0]['global_update']==1
    assert records[0]['run_id']=='run1' and records[0]['attempt_id']=='attempt1'
    assert records[0]['timestamp'].endswith('+00:00')
    assert records[1]['fallbacks']==['invalid_peak']


def test_augmentation_summary_reconciles_clean_augmented_and_fallback_views(tmp_path):
    from asr.logging import RunLogger
    path=tmp_path/'events.jsonl'
    with RunLogger(path,'r','a') as logger:
        logger.log('augmentation',utterance_id='s/1',epoch=0,branch='clean',changed=False,fallbacks=[])
        logger.log('augmentation',utterance_id='s/2',epoch=0,branch='augmentation',changed=False,fallbacks=['speed_overflow','invalid_peak'])
        assert logger.summary['views']==2
        assert logger.summary['clean']==1 and logger.summary['augmentation']==1
        assert logger.summary['fallback_views']==1
    summary=json.loads(path.with_suffix('.summary.json').read_text())
    assert summary['fallbacks']=={'speed_overflow':1,'invalid_peak':1}
