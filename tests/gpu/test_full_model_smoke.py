"""Real allocation-only smoke; it cannot be counted as portable CPU acceptance."""
import os
from pathlib import Path
import socket
import uuid
import pytest

pytestmark=pytest.mark.gpu


def test_full_model_smoke():
    from asr.config import load_config
    from asr.contracts import PreflightReport
    from asr.audit import audit_train
    from asr.paths import read_json
    from asr.precision import require_gpu_allocation
    from asr.smoke import run_smoke
    require_gpu_allocation(os.environ,socket.gethostname())
    config=load_config(Path(os.environ.get('ASR_CONFIG','configs/official.yaml')))
    reports=[PreflightReport.from_dict(read_json(Path(os.environ.get(f'ASR_{stage.upper()}_REPORT',f'manifests/preflight-{stage}.json'))))
             for stage in ['cpu','gpu']]
    report=run_smoke(config,audit_train(config,config.paths.manifest_dir),reports,'smoke-test-'+uuid.uuid4().hex)
    assert report.passed and report.execution=='slurm_gpu'
    comparison = report.metadata['comparison']
    assert comparison['window_sizes'] == [16,16,12]
    assert comparison['global_update'] == 3 and comparison['global_micro_step'] == 44
    assert comparison['continuous_repeat_matched'] and comparison['immediate_restore_exact']
    assert comparison['pre_step_gradients_matched'] and comparison['matched']
    assert (comparison['rtol'],comparison['atol']) == (1e-4,1e-5)
    diagnostic = read_json(Path(comparison['diagnostic']))
    assert diagnostic['determinism']['algorithms'] and not diagnostic['determinism']['warn_only']
    assert diagnostic['determinism']['cudnn_deterministic']
    assert not diagnostic['determinism']['cudnn_benchmark']
