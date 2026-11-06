# PhoWhisper-large LoRA ASR

Implementation of [docs/specs.md](docs/specs.md), with execution order in
[docs/plan.md](docs/plan.md) and evidence in [docs/tasks.md](docs/tasks.md).
T01–T12 provide software and portable tests. UIT commissioning (T13), official
training and complete transcripts (T14) require actual server evidence.

The official method is immutable in `configs/official.yaml`: frozen
`vinai/PhoWhisper-large` at revision `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb`,
encoder/decoder q/v LoRA r=8, E4 augmentation, 10 epochs, batch 1, accumulation 16,
AdamW 1e-4, no validation or deduplication. There are no hyperparameter CLI overrides.
With 5,036 occurrences, training performs 315 updates/epoch and 3,150 total;
the last window contains 12 losses and warmup spans 158 updates. The first
successful optimizer call uses LR=0 and still advances AdamW moments.

## Files and permissions

On Ubuntu, the project is `/datastore/binhvp/BuiTrongTan/ASR`; read-only data is
`/datastore/binhvp/BuiTrongTan/data/train+public_test`. `--data-root` must point
directly to the directory containing `train/` and `public_test/`. Source JSONL,
WAVs and environment directories are never copied into project artifacts.
`CS406/` is outside scope. Outputs, assets, virtual environment and caches stay
under the project. Dataset, checkpoints, weights and transcripts are Git-ignored.

Only CPU preparation belongs on `login01`. GPU preflight, smoke, training,
export and inference require a Slurm compute allocation and one H200 MIG
`2g.35gb` instance. The runtime checks the allocated device and its profile;
cluster GRES spelling is an operational setting, never an assumed partition name.

## Prepare the environment

Obtain the actual account, partition, QoS, MIG GRES, CPU/RAM/walltime and driver
compatibility from the cluster operator. Then, on Ubuntu:

```bash
export ASR_PROJECT_ROOT=/datastore/binhvp/BuiTrongTan/ASR
cd "$ASR_PROJECT_ROOT"
cp configs/slurm.example.env configs/slurm.local.env
# Edit the local file. Empty resource settings intentionally prevent submission.
export ASR_SLURM_ENV="$ASR_PROJECT_ROOT/configs/slurm.local.env"
source "$ASR_SLURM_ENV"
# ASR_TORCH_INDEX_URL must be the explicit official PyTorch CUDA wheel index
# compatible with the actual driver. ASR_PYTHON311 optionally selects Python 3.11.
bash scripts/setup_env.sh
source .venv/bin/activate
```

Setup installs candidate pins from `pyproject.toml` and writes
`requirements/candidate-resolved-linux.txt`. It does not certify CUDA compatibility.
After T13 confirms the exact stack, preserve `python -m pip freeze` as
`requirements/uit-lock.txt` and retain it unchanged across attempts. Each run also
records all installed distribution versions, source hashes and `dependency-lock.txt`.
Changing code or dependencies invalidates prior admission evidence.

## G1.cpu and actual Linux verification

```bash
python -m pytest tests -m "not gpu and not snapshot and not linux" -q
for script in scripts/*.sh scripts/*.slurm; do bash -n "$script" || exit; done
python -m pytest tests/test_slurm_scripts.py tests/test_inference_resume.py \
  tests/test_checkpoint.py -m linux -q
sha256sum scripts/*.sh scripts/*.slurm src/asr/*.py
```

The Linux command exercises real USR1 delivery through a fake synchronous srun,
process termination releasing writer locks, fsynced journal/torn-tail recovery,
and a killed checkpoint writer before publication. Run it on the actual project
filesystem at CPU-only T13.1, saving command output, counts and source/script
hashes. Git Bash syntax/fake-scheduler tests on Windows establish portable
software behavior only. They do not satisfy G1.linux. No required skipped test
counts as a pass. Full G1 and input gate G2 precede GPU commissioning.

## Prepare assets explicitly

This is the only model download step; it requires network access and must be
run deliberately. It downloads the pinned revision under the project, hashes
the inventory, checks architecture and atomically publishes the snapshot.
An existing destination is never overwritten.

```bash
python scripts/prepare_snapshot.py --config configs/official.yaml
```

Runtime loads are local-only with `trust_remote_code=False`. A missing or changed
snapshot fails preflight. Download staging directories remain private if preparation
is interrupted; restart into a new absent destination rather than claiming them complete.

Prepare independently licensed, reviewed non-speech recordings in `assets/noise/`.
Input assets must be PCM mono signed 16-bit 16 kHz, nonempty and positive-power;
long recordings are supported. Preserve the license/source evidence. Create an
ordered `assets/noise/manifest.json`, using actual hashes and review results:

```json
[
  {
    "path": "background.wav",
    "sha256": "REPLACE_WITH_ACTUAL_FILE_SHA256",
    "license": "REPLACE_WITH_VERIFIED_LICENSE",
    "source_url": "REPLACE_WITH_ACTUAL_SOURCE_OR_LICENSE_EVIDENCE",
    "reviewed": false,
    "contains_speech": false
  }
]
```

Set `reviewed: true` only after the real asset has been reviewed. This template
is intentionally invalid. No synthetic noise, dataset audio or silently disabled
noise branch can substitute for the required asset.

## Audit and commission

All commands use the same validated configuration. Operational `--snapshot` and
`--noise-manifest` paths are available if preparation used a different project-local
location; pass the same values throughout the lifecycle.

```bash
asr audit --config configs/official.yaml --data-root "$ASR_DATA_ROOT" \
  --output manifests/train
asr preflight --config configs/official.yaml --data-root "$ASR_DATA_ROOT" \
  --stage cpu --output manifests/preflight-cpu.json
bash scripts/submit_job.sh preflight --output manifests/preflight-gpu.json
# Wait for a passing GPU report, then submit smoke separately:
bash scripts/submit_job.sh smoke --run-id smoke-commission-001 \
  --cpu-report manifests/preflight-cpu.json --gpu-report manifests/preflight-gpu.json
```

Audit requires the pinned Train JSONL hash, 1,545 sources and 5,036 current/user-history
occurrences. It rejects malformed/truncated WAVs, missing references, unsafe paths,
duplicate IDs and same-PCM conflicting transcripts. Duplicate occurrences are retained.
Public Test and assistant turns are excluded. Token lengths are checked using the exact
snapshot before admission; no truncation is permitted.

GPU preflight probes maximum feature/label shape and BF16, otherwise FP16 with
GradScaler. OOM stops the run without tuning the locked method. Smoke uses 16/16/12
windows, a real complete checkpoint, uninterrupted/resumed state comparison and
encoder/decoder gradient checks. Smoke weights never initialize official training.
The passing report is `runs/smoke-commission-001/smoke-report.json`.

Inside the allocated commissioning job, also run the separately marked actual-snapshot
forward test:

```bash
export ASR_CONFIG="$ASR_PROJECT_ROOT/configs/official.yaml"
python -m pytest tests/gpu/test_snapshot_mask_forward.py -m "gpu or snapshot" -q
```

Read the test's required environment paths before running other GPU integration tests.
They are never substituted with tiny-model CPU results.

## Train and resume

Only after matching CPU/GPU/smoke evidence passes and the official run is scheduled:

```bash
bash scripts/submit_job.sh train --run-id official-001 \
  --cpu-report manifests/preflight-cpu.json --gpu-report manifests/preflight-gpu.json \
  --smoke-report runs/smoke-commission-001/smoke-report.json

# After interruption, select the latest complete checkpoint of this same run:
bash scripts/submit_job.sh resume --run-id official-001 \
  --checkpoint runs/official-001/checkpoints/ATTEMPT/update-NNNNNN-KIND
```

Checkpoints preserve adapter, optimizer, scheduler, scaler, all RNG state,
permutation/cursor and identities. They are immutable and published only at clean
window boundaries every 50 updates, at epoch ends and at terminal completion.
Incomplete staging directories cannot resume. Each allocation creates a fresh
attempt UUID, even if Slurm reuses its job ID. Logs and augmentation decisions are
under `runs/<run>/attempts/<attempt>/logs/`; Slurm stdout/stderr are under `runs/slurm/`.

`ASR_SIGNAL_SECONDS` must exceed the measured worst remaining accumulation window,
checkpoint write duration and safety margin. The wrapper requests `B:USR1` for the
batch shell, forwards it through synchronous srun and waits for the task's actual
exit status. Python stops at a clean window and saves. Exit 3 means interrupted;
nonzero errors require investigation. Test actual Slurm delivery and checkpoint lead
time during T13; fake scheduling alone is insufficient. See the official
[sbatch signal](https://slurm.schedmd.com/sbatch.html#OPT_signal) and
[srun signal forwarding](https://slurm.schedmd.com/srun.html#SECTION_SIGNALS-AND-ESCAPE-SEQUENCES)
documentation. Jobs never submit another stage automatically.

## Export and transcribe original Train audio

After the terminal checkpoint proves 10 epochs, 3,150 successful updates and 50,360
micro-steps, run export in a compute allocation:

```bash
bash scripts/submit_job.sh export \
  --checkpoint runs/official-001/checkpoints/ATTEMPT/update-003150-terminal \
  --output runs/official-001/merged-fp16

bash scripts/submit_job.sh infer \
  --model runs/official-001/merged-fp16 \
  --manifest runs/official-001/manifests/utterances.jsonl \
  --output runs/official-001/transcripts

asr validate-output --config configs/official.yaml \
  --model runs/official-001/merged-fp16 \
  --manifest runs/official-001/manifests/utterances.jsonl \
  --utterances runs/official-001/transcripts/utterances.jsonl \
  --sources runs/official-001/transcripts/train_transcripts.jsonl
```

Export merges in FP32, converts floating weights to FP16, checks logits at fixed
`atol=0.05, rtol=0.05` and verifies fresh-process offline reload and short generation.
Its technical one-token reload probe does not change the exported 440-token inference
configuration. Publication never overwrites an existing export.

Inference uses original WAVs, eval/inference mode, Vietnamese transcription, greedy
decoding and a 440-token limit. Empty/error/no-EOS outputs stop with diagnostic token
evidence; references never replace predictions. Repeat the **same infer command** after
interruption. Its `.inference-progress/` journal resumes only a matching model/config/
manifest/generation identity and validates every committed prefix row. It can recover
an unterminated final fragment; a corrupt complete line or different identity fails.

Final files contain 5,036 utterance predictions and 1,545 source records, preserving
original current/history IDs and history indices. Consumers must require a valid
`output-validation.json` whose file hashes match both public files. The report is
published last; a matching interrupted publication can finish idempotently.
Run manifests/summaries distinguish training, export and transcript completion.

## Failure handling

Missing assets/allocation, identity mismatch, dataset conflicts, non-finite gradients,
OOM and skipped scaler updates are explicit failures. Preserve diagnostics and the
last complete checkpoint, address the cause, then resume the same logical run.
Do not change rank, batch, precision, data order, augmentation or epoch budget to
work around a failure. A method change needs a new explicit decision and run identity.
G2–G6 and actual Linux/GPU checks remain pending until their real evidence is recorded.
