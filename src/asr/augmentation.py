"""One reproducible E4 view per occurrence/epoch, without global RNG state."""
import hashlib
import numpy as np

from .assets import load_noise
from .contracts import AugmentationResult


def augmentation_seed(seed: int, epoch: int, utterance_id: str) -> int:
    material = f'{seed}:{epoch}:{utterance_id}:augmentation-v1'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(material).digest()[:8], 'little', signed=False)


def augment(audio, row, epoch, noise, seed=42) -> AugmentationResult:
    clean = np.asarray(audio, dtype=np.float32).copy()
    if clean.ndim != 1 or not 0 < len(clean) <= 480000 or not np.isfinite(clean).all():
        raise ValueError('invalid clean waveform')
    if not noise.entries:
        raise ValueError('E4 policy requires verified nonempty noise assets')
    local_seed = augmentation_seed(seed,epoch,row.utterance_id)
    rng = np.random.default_rng(local_seed)
    decisions, fallbacks = {}, []
    def result(wave, branch):
        return AugmentationResult(wave,branch,decisions,fallbacks,row.utterance_id,epoch,
                                  local_seed,not np.array_equal(wave,clean))
    if rng.random() >= 0.5:
        return result(clean,'clean')
    wave = clean.copy()
    decisions['speed_selected'] = bool(rng.random() < 0.5)
    if decisions['speed_selected']:
        factor = float(rng.choice([0.9,1.0,1.1]))
        count = round(len(wave)/factor)
        if count > 480000:
            fallbacks.append('speed_overflow')
            factor,count = 1.0,len(wave)
        decisions['speed_factor'] = factor
        if factor != 1.0:
            wave = np.interp(np.linspace(0,len(wave)-1,count),np.arange(len(wave)),wave).astype(np.float32)
    decisions['volume_selected'] = bool(rng.random() < 0.5)
    if decisions['volume_selected']:
        gain_db = float(rng.uniform(-6,6))
        gain = 10 ** (gain_db/20)
        peak = float(np.max(np.abs(wave))) if len(wave) else 0.0
        if peak:
            gain = min(gain,0.99/peak)
        wave = (wave.astype(np.float64)*gain).astype(np.float32)
        decisions.update(gain_db=gain_db,effective_gain=gain)
    decisions['noise_selected'] = bool(rng.random() < 0.5)
    if decisions['noise_selected']:
        index = int(rng.integers(0,len(noise.entries)))
        asset = load_noise(noise.entries[index])
        offset = int(rng.integers(0,len(asset)))
        snr = float(rng.uniform(15,30))
        decisions.update(noise_index=index,noise_offset=offset,snr_db=snr)
        segment = asset[(offset+np.arange(len(wave))) % len(asset)].astype(np.float64)
        signal = wave.astype(np.float64)
        noise_power,signal_power = np.mean(segment**2),np.mean(signal**2)
        if noise_power <= 0:
            fallbacks.append('noise_zero_power')
            return result(clean,'augmentation')
        scale = np.sqrt(signal_power/(noise_power*10**(snr/10)))
        wave = (signal+scale*segment).astype(np.float32)
    reason = None
    if wave.size == 0:
        reason = 'empty_waveform'
    elif len(wave) > 480000:
        reason = 'duration_overflow'
    elif not np.isfinite(wave).all():
        reason = 'non_finite_waveform'
    elif np.max(np.abs(wave)) >= 1:
        reason = 'invalid_peak'
    if reason:
        fallbacks.append(reason)
        wave = clean
    return result(wave,'augmentation')
