"""Source-to-utterance projection, retaining the original history positions."""
from pathlib import Path
import json
import hashlib
import struct
import numpy as np

from .contracts import UtteranceRow


def project_sources(train_jsonl: Path) -> list[UtteranceRow]:
    rows, seen = [], set()
    with Path(train_jsonl).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                sid = item.get("id")
                if not isinstance(sid, str) or not sid.strip() or "/" in sid or "\\" in sid:
                    raise ValueError("invalid source ID")
                if sid in seen:
                    raise ValueError(f"duplicate source ID: {sid}")
                seen.add(sid)
                history = item.get("history", [])
                if not isinstance(history, list):
                    raise ValueError(f"{sid}: history must be a list")
                turns = [("current", item, None)]
                for index, turn in enumerate(history):
                    if not isinstance(turn, dict):
                        raise ValueError(f"{sid}/history_{index:03d}: invalid turn")
                    if turn.get("role") == "user":
                        turns.append((f"history_{index:03d}", turn, index))
                for turn_id, turn, index in turns:
                    audio, reference = turn.get("audio"), turn.get("audio_text")
                    if not isinstance(audio, str) or not audio.strip() or not isinstance(reference, str):
                        raise ValueError(f"{sid}/{turn_id}: missing audio or transcript")
                    rows.append(UtteranceRow(sid, turn_id, f"{sid}/{turn_id}", audio, reference, index))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
    return rows


def read_pcm16(path: Path) -> tuple[np.ndarray, dict]:
    """Validate RIFF payload lengths before interpreting fixed-format PCM."""
    raw = Path(path).read_bytes()
    if len(raw) < 12 or raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
        raise ValueError('invalid WAV header')
    declared = struct.unpack_from('<I', raw, 4)[0] + 8
    if declared > len(raw):
        raise ValueError('truncated WAV payload')
    offset, fmt, pcm = 12, None, None
    while offset + 8 <= declared:
        kind = raw[offset:offset+4]
        length = struct.unpack_from('<I', raw, offset+4)[0]
        start = offset + 8
        if start + length > declared:
            raise ValueError('truncated WAV chunk')
        if kind == b'fmt ':
            if fmt is not None or length < 16:
                raise ValueError('invalid WAV format chunk')
            fmt = struct.unpack_from('<HHIIHH', raw, start)
        elif kind == b'data':
            if pcm is not None:
                raise ValueError('multiple PCM payloads are unsupported')
            pcm = raw[start:start+length]
        offset = start + length + length % 2
    if fmt is None or pcm is None:
        raise ValueError('missing WAV format or data')
    if fmt != (1, 1, 16000, 32000, 2, 16) or len(pcm) % 2:
        raise ValueError('audio format must be PCM mono 16-bit 16 kHz')
    frames = len(pcm) // 2
    if not 0 < frames <= 480000:
        raise ValueError('audio duration must be >0 and <=30 seconds')
    audio = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
    if not np.isfinite(audio).all():
        raise ValueError('non-finite waveform')
    return audio, {'wav_sha256':hashlib.sha256(raw).hexdigest(),
                   'pcm_sha256':hashlib.sha256(pcm).hexdigest(),
                   'duration_seconds':frames/16000, 'num_frames':frames,
                   'sample_rate':16000,'channels':1,'sample_width_bytes':2}
