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


@dataclass(frozen=True)
class ArtifactIdentity:
    method_hash: str
    dataset_hash: str
    manifest_hash: str
    snapshot_hash: str
    noise_hash: str
    runtime_hash: str
    dependency_hash: str
    base_revision: str = 'b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb'

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def identity_hash(self) -> str:
        from .config import canonical_hash
        return canonical_hash(self.as_dict())


class ReportRecord:
    def as_dict(self) -> dict:
        return {'schema_version':1,**asdict(self)}

    @classmethod
    def from_dict(cls,value):
        raw = dict(value)
        if raw.pop('schema_version',None) != 1:
            raise ValueError('unsupported report schema')
        try:
            if raw.get('identity') is not None:
                raw['identity'] = ArtifactIdentity(**raw['identity'])
            if raw.get('runtime') is not None:
                raw['runtime'] = RuntimeProfile(**raw['runtime'])
            return cls(**raw)
        except (TypeError,KeyError) as exc:
            raise ValueError('invalid report structure') from exc


@dataclass
class PreflightReport(ReportRecord):
    stage: str
    passed: bool
    identity: ArtifactIdentity | None
    checks: list[dict]
    runtime: RuntimeProfile | None = None
    execution: str = 'cpu'
    metadata: dict = field(default_factory=dict)


@dataclass
class SmokeReport(ReportRecord):
    passed: bool
    identity: ArtifactIdentity
    runtime: RuntimeProfile
    fixture_ids: list[str]
    checks: list[dict]
    execution: str = 'unverified'
    metadata: dict = field(default_factory=dict)


@dataclass
class ExportResult(ReportRecord):
    path: Path
    valid: bool
    identity: ArtifactIdentity
    files: dict[str, str]
    metadata: dict = field(default_factory=dict)


@dataclass
class OutputValidation(ReportRecord):
    valid: bool
    utterance_count: int
    source_count: int
    identity: ArtifactIdentity
    files: dict[str, str]
    errors: list[str] = field(default_factory=list)
