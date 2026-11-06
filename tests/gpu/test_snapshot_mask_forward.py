"""Actual snapshot verification: intentionally excluded from portable CPU evidence."""
import os
from pathlib import Path
import numpy as np
import pytest

pytestmark=[pytest.mark.gpu,pytest.mark.snapshot]


def test_real_snapshot_mask_forward():
    import torch
    from transformers import WhisperForConditionalGeneration
    from asr.config import load_config
    from asr.assets import verify_snapshot
    from asr.precision import require_gpu_allocation
    from asr.processor import load_processor,prepare_example
    from asr.contracts import UtteranceRow
    require_gpu_allocation(os.environ,os.environ.get('HOSTNAME',''))
    config=load_config(Path(os.environ.get('ASR_CONFIG','configs/official.yaml')))
    snapshot=verify_snapshot(config)
    processor=load_processor(snapshot)
    example=prepare_example(UtteranceRow('probe','current','probe/current','audio/probe.wav','Kiểm tra.'),
                            np.zeros(16000,dtype=np.float32),processor)
    model=WhisperForConditionalGeneration.from_pretrained(snapshot.path,local_files_only=True,
                                                         trust_remote_code=False,torch_dtype=torch.float32).cuda().eval()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):
        result=model(**{k:v.cuda() for k,v in example.model_inputs().items()})
    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.logits).all()
