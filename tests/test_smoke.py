import pytest
import torch


def test_smoke_harness_preserves_three_windows_and_resumes_real_optimizer_state(local_config):
    from training_helpers import ToyLoss,examples,bundle_for
    from test_checkpoint import identity_fixture
    from asr.checkpoint import TrainingObjects
    from asr.model import seed_all
    from asr.training import create_optimizer,compute_budget
    from asr.schedule import create_scheduler
    from asr.smoke import compare_resume
    def factory():
        seed_all(42)
        bundle=bundle_for(ToyLoss()); opt=create_optimizer(bundle,local_config)
        return TrainingObjects(bundle,opt,create_scheduler(opt,compute_budget(5036)),None)
    result=compare_resume(factory,examples(44),identity_fixture(),local_config,
                          local_config.paths.project_root/'runs/smoke-test',list(range(5036)))
    assert result['matched'] is True
    assert result['window_sizes']==[16,16,12]
    assert result['global_update']==3 and result['global_micro_step']==44
    assert result['initial_adapter_hash']==result['resumed_initial_adapter_hash']


def test_smoke_entry_refuses_cpu_fixture_as_actual_gpu_evidence(local_config,monkeypatch):
    from asr.smoke import run_smoke
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='allocation'):
        run_smoke(local_config,None,[],'smoke-refused')
    assert not (local_config.paths.project_root/'runs/smoke-refused').exists()


def test_smoke_cli_reports_missing_evidence_with_nonzero_exit(tmp_path):
    from asr.cli import main
    assert main(['smoke','--run-id','smoke-cli','--cpu-report',str(tmp_path/'cpu.json'),
                 '--gpu-report',str(tmp_path/'gpu.json')])==1
