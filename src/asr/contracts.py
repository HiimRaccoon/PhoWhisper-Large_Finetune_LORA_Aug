"""Versioned records shared by pipeline stages; runtime tensors are never JSON fields."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class UtteranceRow:
    sample_id: str
    turn_id: str
    utterance_id: str
    audio: str
    reference: str
    history_index: int | None = None
    wav_sha256: str = ""
    pcm_sha256: str = ""
    duration_seconds: float = 0.0
    num_frames: int = 0
    sample_rate: int = 16000
    channels: int = 1
    sample_width_bytes: int = 2
    normalized_reference: str = ""
    label_token_count: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditResult:
    valid: bool
    source_count: int
    utterance_count: int
    manifest_path: Path
    report_path: Path
    source_sha256: str
    manifest_sha256: str
    errors: list[dict]
    duplicate_groups: int = 0
    conflict_groups: int = 0

    def as_dict(self) -> dict:
        result = asdict(self)
        result['schema_version'] = SCHEMA_VERSION
        return result
