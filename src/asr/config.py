"""Immutable, strictly validated official methodology and separate operational paths."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

BASE_REVISION = "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb"
TRAIN_SHA256 = "24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8"
LOCKED = {
    "model": {"base": "vinai/PhoWhisper-large", "revision": BASE_REVISION, "frozen_base": True,
              "quantization": None, "trust_remote_code": False},
    "data": {"source_samples": 1545, "expected_utterances": 5036, "include_current": True,
             "include_user_history": True, "include_assistant": False, "validation": False, "deduplicate": False},
    "lora": {"scope": "encoder_decoder", "rank": 8, "alpha": 16, "dropout": 0.05,
             "bias": "none", "modules_to_save": None},
    "training": {"seed": 42, "epochs": 10, "micro_batch_size": 1, "gradient_accumulation_steps": 16,
                 "optimizer": "AdamW", "learning_rate": 0.0001, "betas": [0.9, 0.999], "epsilon": 1.0e-8,
                 "weight_decay": 0.01, "scheduler": "linear", "warmup_ratio": 0.05, "max_grad_norm": 1.0,
                 "gradient_checkpointing": True, "use_cache": False, "precision": "auto",
                 "checkpoint_every_updates": 50, "early_stopping": False},
    "augmentation": {"policy": "augmentation-v1", "branch_probability": 0.5, "speed_probability": 0.5,
                     "speed_factors": [0.9, 1.0, 1.1], "volume_probability": 0.5, "gain_db": [-6.0, 6.0],
                     "noise_probability": 0.5, "snr_db": [15.0, 30.0]},
    "processor": {"sample_rate": 16000, "mel_bins": 80, "language": "vi", "task": "transcribe",
                  "predict_timestamps": False, "max_label_tokens": 447},
    "inference": {"offline": True, "do_sample": False, "language": "vi", "task": "transcribe",
                  "num_beams": 1, "max_new_tokens": 440},
    "export": {"merge_adapter": True, "dtype": "float16"},
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check(actual: Any, expected: Any, name: str = "") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{name or 'configuration'} has missing or unknown fields")
        for key, value in expected.items():
            _check(actual[key], value, f"{name}.{key}".strip("."))
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{name} must remain locked at {expected!r}")


class Section:
    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]):
        object.__setattr__(self, "_values", MappingProxyType({
            k: tuple(v) if isinstance(v, list) else v for k, v in values.items()
        }))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Official methodology is immutable")

    def as_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self._values.items()}


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_root: Path
    snapshot: Path
    noise_manifest: Path

    @property
    def train_root(self) -> Path:
        return self.data_root / "train"

    @property
    def manifest_dir(self) -> Path:
        return self.project_root / "manifests/train"


@dataclass(frozen=True)
class OfficialConfig:
    sections: Mapping[str, Section]
    paths: Paths

    def __getattr__(self, name: str) -> Section:
        try:
            return self.sections[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def method_dict(self) -> dict:
        return {name: section.as_dict() for name, section in self.sections.items()}

    @property
    def method_hash(self) -> str:
        return canonical_hash(self.method_dict())

    def resolved_dict(self) -> dict:
        return {**self.method_dict(), "paths": {
            key: str(value) for key, value in vars(self.paths).items()
        }}

    def with_paths(self, **changes: Any) -> OfficialConfig:
        raw = {key: str(value) for key, value in vars(self.paths).items()}
        raw.update(changes)
        return replace(self, paths=_paths(raw, self.paths.project_root))


def _paths(raw: dict, default_root: Path) -> Paths:
    allowed = {"project_root", "data_root", "snapshot", "noise_manifest"}
    if set(raw) - allowed:
        raise ValueError("paths has unknown fields")
    for key, value in raw.items():
        if value is not None and not isinstance(value, (str, Path)):
            raise ValueError(f"paths.{key} must be a path")
    root = Path(raw.get("project_root") or default_root).resolve()
    if any(part.casefold() == "cs406" for part in root.parts):
        raise ValueError("CS406 is outside this project")
    def local(key: str, default: str) -> Path:
        path = Path(raw.get(key) or default)
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"paths.{key} must stay under project_root")
        if any(part.casefold() == "cs406" for part in path.parts):
            raise ValueError("CS406 is outside this project")
        return path
    return Paths(root, Path(raw.get("data_root") or "/datastore/binhvp/BuiTrongTan/data/train+public_test").resolve(),
                 local("snapshot", "assets/pretrained/PhoWhisper-large"),
                 local("noise_manifest", "assets/noise/manifest.json"))


def validate_locked(config: OfficialConfig) -> None:
    _check(config.method_dict(), LOCKED)


def load_config(path: Path) -> OfficialConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    method = {key: value for key, value in raw.items() if key != "paths"}
    _check(method, LOCKED)
    operational = raw.get("paths", {})
    if not isinstance(operational, dict):
        raise ValueError("paths must be a mapping")
    return OfficialConfig(MappingProxyType({key: Section(value) for key, value in method.items()}),
                          _paths(operational, path.parent.parent))
