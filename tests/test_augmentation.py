from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest


class Draws:
    def __init__(self,gates,factor=0.9,uniform=20,indices=(0,0)):
        self.gates=iter(gates); self.factor=factor; self.value=uniform; self.indices=iter(indices)
    def random(self): return next(self.gates)
    def choice(self,values): return self.factor
    def uniform(self,low,high):
        assert low <= self.value <= high
        return self.value
    def integers(self,low,high):
        value=next(self.indices); assert low <= value < high; return value


def noise_fixture(config,make_wav,samples=(100,-100,100,-100)):
    from asr.assets import verify_noise
    root=config.paths.noise_manifest.parent
    path=make_wav(root/'noise.wav',samples=samples)
    entry={'path':'noise.wav','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'license':'CC0',
           'reviewed':True,'contains_speech':False,'source_url':'https://example.invalid/fixture'}
    config.paths.noise_manifest.write_text(json.dumps([entry]),encoding='utf-8')
    return verify_noise(config)


def row_fixture():
    from asr.contracts import UtteranceRow
    return UtteranceRow('train_0001','current','train_0001/current','audio/x.wav','Reference')


def test_seed_and_view_replay_are_deterministic(local_config,make_wav):
    from asr.augmentation import augmentation_seed,augment
    noise=noise_fixture(local_config,make_wav)
    assert augmentation_seed(42,0,'train_0001/current')==18138155238159038922
    audio=np.linspace(-0.2,0.2,1000,dtype=np.float32)
    first=augment(audio,row_fixture(),0,noise)
    repeat=augment(audio,row_fixture(),0,noise)
    np.testing.assert_array_equal(first.audio,repeat.audio)
    assert first.decisions==repeat.decisions
    assert row_fixture().reference=='Reference'


@pytest.mark.parametrize('factor,frames',[(0.9,1111),(1.0,1000),(1.1,909)])
def test_speed_gate_keeps_sample_rate_and_expected_frame_count(local_config,make_wav,monkeypatch,factor,frames):
    from asr.augmentation import augment
    noise=noise_fixture(local_config,make_wav)
    monkeypatch.setattr(np.random,'default_rng',lambda seed:Draws([0.1,0.1,0.9,0.9],factor=factor))
    assert len(augment(np.full(1000,0.1,dtype=np.float32),row_fixture(),0,noise).audio)==frames


def test_clean_branch_does_not_draw_or_modify_further(local_config,make_wav,monkeypatch):
    from asr.augmentation import augment
    noise=noise_fixture(local_config,make_wav)
    monkeypatch.setattr(np.random,'default_rng',lambda seed:Draws([0.9]))
    audio=np.full(100,0.2,dtype=np.float32)
    result=augment(audio,row_fixture(),0,noise)
    assert result.branch=='clean'
    np.testing.assert_array_equal(result.audio,audio)


def test_volume_is_capped_and_noise_wrap_has_requested_snr(local_config,make_wav,monkeypatch):
    from asr.augmentation import augment
    noise=noise_fixture(local_config,make_wav)
    monkeypatch.setattr(np.random,'default_rng',lambda seed:Draws([0.1,0.9,0.1,0.9],uniform=6))
    volume=augment(np.full(8,0.8,dtype=np.float32),row_fixture(),0,noise)
    assert np.max(np.abs(volume.audio))==pytest.approx(0.99)
    monkeypatch.setattr(np.random,'default_rng',lambda seed:Draws([0.1,0.9,0.9,0.1],uniform=20,indices=(0,3)))
    signal=np.full(8,0.1,dtype=np.float32)
    mixed=augment(signal,row_fixture(),0,noise)
    np.testing.assert_allclose(mixed.audio,[0.09,0.11]*4,atol=1e-7)
    snr=10*np.log10(np.mean(signal.astype(float)**2)/np.mean((mixed.audio-signal).astype(float)**2))
    assert snr==pytest.approx(20,abs=1e-4)


def test_overflow_and_zero_power_fallbacks_are_logged(local_config,make_wav,monkeypatch):
    from asr.augmentation import augment
    noise=noise_fixture(local_config,make_wav,samples=(0,0,100,-100))
    monkeypatch.setattr(np.random,'default_rng',lambda seed:Draws([0.1,0.1,0.9,0.9]))
    result=augment(np.full(480000,0.1,dtype=np.float32),row_fixture(),0,noise)
    assert len(result.audio)==480000 and 'speed_overflow' in result.fallbacks
    monkeypatch.setattr(np.random,'default_rng',lambda seed:Draws([0.1,0.9,0.9,0.1],uniform=20))
    audio=np.full(2,0.1,dtype=np.float32)
    result=augment(audio,row_fixture(),0,noise)
    np.testing.assert_array_equal(result.audio,audio)
    assert 'noise_zero_power' in result.fallbacks


def test_independent_zero_signal_golden_fixture(local_config,make_wav):
    from pathlib import Path
    from asr.augmentation import augment
    fixture=json.loads((Path(__file__).parent/'fixtures/augmentation-v1.json').read_text())
    noise=noise_fixture(local_config,make_wav)
    result=augment(np.zeros(8,dtype=np.float32),row_fixture(),0,noise)
    assert hashlib.sha256(result.audio.tobytes()).hexdigest()==fixture['zero_signal_case']['expected_sha256']


def test_noise_clipping_falls_back_without_changing_reference(local_config,make_wav,monkeypatch):
    from asr.augmentation import augment
    noise=noise_fixture(local_config,make_wav)
    monkeypatch.setattr(np.random,'default_rng',lambda seed:Draws([0.1,0.9,0.9,0.1],uniform=20))
    original=np.full(8,0.999,dtype=np.float32)
    result=augment(original,row_fixture(),0,noise)
    np.testing.assert_array_equal(result.audio,original)
    assert result.fallbacks==['invalid_peak']
