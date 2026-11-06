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
