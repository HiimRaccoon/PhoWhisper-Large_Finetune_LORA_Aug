"""An ordinary git add must never stage runtime assets or competition audio."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generated_artifacts_are_not_tracked_but_source_templates_are(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
    ignored = ["assets/pretrained/model.safetensors", "model.bin", "runs/r/checkpoint/adapter.pt",
               "outputs/transcripts.jsonl", "train+public_test/train/audio.wav", "copy.wav",
               ".venv/lib/x", ".cache/x", ".tmp-checkpoint-1/weights", "configs/server.local.yaml",
               ".env", "assets/noise/background.wav", "manifests/audit.json"]
    trackable = ["src/asr/config.py", "configs/official.yaml", "configs/slurm.example.env",
                 "tests/test_config.py", "docs/specs.md", "requirements/uit-lock.txt", ".env.example"]
    for name in ignored + trackable:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "--all"], check=True)
    tracked = subprocess.check_output(["git", "-C", str(tmp_path), "ls-files"], text=True).splitlines()
    assert set(trackable).issubset(tracked)
    assert set(ignored).isdisjoint(tracked)
