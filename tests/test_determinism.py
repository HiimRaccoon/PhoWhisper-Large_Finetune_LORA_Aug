"""Runtime reproducibility is independent of the locked training hyperparameters."""
import pytest
import torch


def test_deterministic_policy_is_strict_rng_neutral_and_preserves_precision(monkeypatch):
    from asr.precision import configure_determinism
    from asr.checkpoint import capture_rng
    from asr.smoke import _compare
    before = capture_rng()
    precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    try:
        configure_determinism()
        configure_determinism()  # Idempotent; never reseed on entry/resume.
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        import os
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
        assert precision == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        _compare(before, capture_rng())
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic


@pytest.mark.parametrize('workspace', [None, 'invalid'])
def test_policy_rejects_late_or_invalid_workspace(monkeypatch, workspace):
    from asr.precision import configure_determinism
    if workspace is None:
        monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    else:
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', workspace)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    with pytest.raises(RuntimeError, match='CUBLAS_WORKSPACE_CONFIG'):
        configure_determinism()
