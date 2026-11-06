import pytest


@pytest.mark.parametrize('env,host',[({},'compute01'),({'SLURM_JOB_ID':'1'},'login01'),
                                     ({'SLURM_JOB_ID':'1'},'login01.example')])
def test_gpu_allocation_guard_rejects_before_cuda(env,host):
    from asr.precision import require_gpu_allocation
    with pytest.raises(RuntimeError,match='allocation|login'):
        require_gpu_allocation(env,host)


def test_precision_probe_uses_bf16_then_fp16_without_hiding_oom():
    import torch
    from asr.precision import select_precision
    calls=[]
    def probe(dtype):
        calls.append(dtype)
        if dtype=='bfloat16': raise RuntimeError('kernel unsupported')
        return {'peak_bytes':100}
    result=select_precision(True,probe)
    assert result.compute_dtype=='float16' and result.scaler_enabled
    assert calls==['bfloat16','float16']
    def oom(dtype): raise torch.OutOfMemoryError('out of memory')
    with pytest.raises(torch.OutOfMemoryError): select_precision(True,oom)
