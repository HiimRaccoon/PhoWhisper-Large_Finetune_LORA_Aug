"""Filesystem containment, hashing and durable small artifact writes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import uuid


def _forbidden(parts) -> bool:
    return any(p.casefold() == "cs406" or p.casefold().endswith(("_environment", "_enviroment")) for p in parts)


def resolve_beneath(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    raw = str(relative).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute() or PureWindowsPath(raw).drive or ".." in path.parts or _forbidden(root.parts + path.parts):
        raise ValueError(f"unsafe path: {relative}")
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"path escapes root: {relative}")
    return result


def project_path(config, path: Path) -> Path:
    root = config.paths.project_root
    path = Path(path)
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not candidate.is_relative_to(root) or _forbidden(candidate.parts):
        raise ValueError("output path must stay under project_root")
    return candidate


def audio_path(train_root: Path, audio: str) -> Path:
    raw = str(audio).replace("\\", "/")
    if not raw.startswith("audio/"):
        raise ValueError("audio must be under train/audio")
    return resolve_beneath(Path(train_root) / "audio", raw[6:])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, value) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                  allow_nan=False, default=str) + "\n").encode("utf-8"))


def write_jsonl_atomic(path: Path, rows) -> None:
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str) + "\n" for r in rows)
    atomic_bytes(path, payload.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]
