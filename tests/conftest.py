"""Local test imports; all generated fixtures stay inside the project cache."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import wave
import numpy as np
import pytest


@pytest.fixture
def make_wav():
    def write(path, samples=(100, 200, -300, 400), rate=16000, channels=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), 'wb') as f:
            f.setnchannels(channels)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(np.asarray(samples, dtype='<i2').tobytes())
        return path
    return write


@pytest.fixture
def local_config(tmp_path):
    from asr.config import load_config
    return load_config(ROOT / 'configs/official.yaml').with_paths(
        project_root=tmp_path, data_root=tmp_path/'dataset',
        snapshot=tmp_path/'assets/pretrained/base', noise_manifest=tmp_path/'assets/noise/manifest.json')


@pytest.fixture
def tiny_whisper():
    import torch
    from transformers import WhisperConfig,WhisperForConditionalGeneration
    torch.set_num_threads(1)
    config=WhisperConfig(vocab_size=64,num_mel_bins=80,d_model=16,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,
        max_source_positions=1500,max_target_positions=448,pad_token_id=0,bos_token_id=1,eos_token_id=2,
        decoder_start_token_id=1,dropout=0.0,attention_dropout=0.0,activation_dropout=0.0,
        encoder_layerdrop=0.0,decoder_layerdrop=0.0,apply_spec_augment=False,
        suppress_tokens=[],begin_suppress_tokens=[])
    return WhisperForConditionalGeneration(config)
