"""Execute only after T14 creates a genuine final export in an allocated GPU job."""
import os
from pathlib import Path
import socket
import pytest


@pytest.mark.gpu
def test_final_export_offline_reload():
    from asr.precision import require_gpu_allocation
    from asr.export import load_export,verify_local_export
    require_gpu_allocation(os.environ,socket.gethostname())
    path=Path(os.environ['ASR_FINAL_EXPORT'])
    assert load_export(path).valid
    assert verify_local_export(path,'cuda')['valid']
