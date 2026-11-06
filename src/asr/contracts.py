"""Versioned records shared by pipeline stages; runtime tensors are never JSON fields."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class SnapshotIdentity:
    path: Path
    repository: str
    revision: str
    files: dict[str, str]
    identity_hash: str


@dataclass(frozen=True)
class NoiseEntry:
    path: Path
    sha256: str
    pcm_sha256: str
    license: str
    source_url: str
    num_frames: int


@dataclass(frozen=True)
class NoiseManifest:
    path: Path
    sha256: str
    entries: tuple[NoiseEntry, ...]


@dataclass
class PreparedExample:
    input_features: Any
    attention_mask: Any
    labels: Any
    utterance_id: str

    def model_inputs(self) -> dict:
        return {'input_features':self.input_features,'attention_mask':self.attention_mask,'labels':self.labels}


@dataclass
class AugmentationResult:
    audio: Any
    branch: str
    decisions: dict
    fallbacks: list[str]
    utterance_id: str
    epoch: int
    seed: int
    changed: bool

    def as_log(self) -> dict:
        return {k:v for k,v in vars(self).items() if k != 'audio'}


@dataclass(frozen=True)
class RuntimeProfile:
    device: str = 'cpu'
    compute_dtype: str = 'float32'
    scaler_enabled: bool = False
    probe: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingBudget:
    utterances_per_epoch: int
    updates_per_epoch: int
    total_updates: int
    warmup_updates: int


@dataclass
class TrainingProgress:
    epoch: int = 0
    completed_epochs: int = 0
    next_position: int = 0
    global_micro_step: int = 0
    global_update: int = 0
    permutation_hash: str = ''
    pending_micro_steps: int = 0
    successful_updates_by_epoch: list[int] = field(default_factory=lambda:[0]*10)
    utterances_completed_by_epoch: list[int] = field(default_factory=lambda:[0]*10)

    def as_dict(self) -> dict:
        return asdict(self)
