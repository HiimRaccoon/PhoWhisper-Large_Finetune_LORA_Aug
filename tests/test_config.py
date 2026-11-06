"""Reject a changed training method rather than silently running another experiment."""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_official_config_loads_and_rejects_changed_epoch_budget(tmp_path):
    from asr.config import load_config

    config = load_config(ROOT / "configs/official.yaml")
    assert config.training.epochs == 10
    assert config.lora.scope == "encoder_decoder"
    raw = yaml.safe_load((ROOT / "configs/official.yaml").read_text(encoding="utf-8"))
    raw["training"]["epochs"] = 9
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="training.epochs"):
        load_config(changed)


@pytest.mark.parametrize("section,key,value", [
    ("lora", "scope", "decoder"), ("lora", "rank", True),
    ("data", "deduplicate", True), ("training", "micro_batch_size", 2),
    ("training", "early_stopping", True), ("inference", "num_beams", 2),
    ("model", "revision", "main"), ("training", "extra", 1),
])
def test_locked_config_rejects_method_changes(tmp_path, section, key, value):
    from asr.config import load_config
    raw = yaml.safe_load((ROOT / "configs/official.yaml").read_text(encoding="utf-8"))
    raw[section][key] = value
    path = tmp_path / "method.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_operational_paths_do_not_change_method_hash(tmp_path):
    from asr.config import load_config
    config = load_config(ROOT / "configs/official.yaml")
    changed = config.with_paths(data_root=tmp_path / "external-dataset")
    assert changed.method_hash == config.method_hash
    with pytest.raises(AttributeError):
        changed.training.epochs = 1
    with pytest.raises(ValueError, match="project_root"):
        config.with_paths(snapshot=ROOT.parent / "outside-asr")
