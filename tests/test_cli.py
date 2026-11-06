from pathlib import Path


def test_cli_audit_reports_invalid_dataset_and_exits_nonzero(tmp_path,capsys):
    from asr.cli import main
    root=Path(__file__).resolve().parents[1]
    code=main(['audit','--config',str(root/'configs/official.yaml'),'--data-root',str(tmp_path/'missing'),
               '--output',str(tmp_path/'report')])
    assert code==1
    assert (tmp_path/'report/audit.json').is_file()


def test_cli_train_requires_all_admission_reports(tmp_path,capsys):
    import pytest
    from asr.cli import main
    with pytest.raises(SystemExit) as error:
        main(['train','--run-id','unverified'])
    assert error.value.code==2


def test_cli_cannot_train_with_fabricated_gpu_reports_without_allocation(tmp_path,monkeypatch,capsys):
    from dataclasses import replace
    from asr.cli import main
    from asr.config import load_config
    from asr.contracts import PreflightReport,SmokeReport,RuntimeProfile
    from asr.paths import write_json_atomic
    from test_checkpoint import identity_fixture
    root=Path(__file__).resolve().parents[1]; cfg=load_config(root/'configs/official.yaml')
    identity=replace(identity_fixture(),method_hash=cfg.method_hash)
    runtime=RuntimeProfile('cuda','bfloat16',False)
    reports={'cpu':PreflightReport('cpu',True,replace(identity,runtime_hash=''),[]),
             'gpu':PreflightReport('gpu',True,identity,[],runtime,execution='slurm_gpu'),
             'smoke':SmokeReport(True,identity,runtime,['s/current'],[],execution='slurm_gpu')}
    args=['train','--run-id','must-not-start','--config',str(root/'configs/official.yaml')]
    for name,report in reports.items():
        path=tmp_path/f'{name}.json'; write_json_atomic(path,report.as_dict())
        args.extend(['--'+name+'-report',str(path)])
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    assert main(args)==1
    assert 'allocation' in capsys.readouterr().err
    assert not (root/'runs/must-not-start').exists()
