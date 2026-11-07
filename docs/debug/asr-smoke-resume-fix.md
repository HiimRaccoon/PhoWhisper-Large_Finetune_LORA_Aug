# ASR smoke/resume source repair and UIT verification

Prepared 2026-11-07, Asia/Ho_Chi_Minh. Local checkout:
`D:\JASR\PhoWhisper-Large_Finetune_LORA_Aug`, starting at
`db2d7205e5e5578f10d0cc4d8e214756c4da8547`.

## Evidence and root-cause boundary

The original `asr-smoke-resume-diagnostic.md` is preserved unchanged. Its logs
establish exact checkpoint restoration of adapter, optimizer, scheduler and
Torch/CUDA RNG, followed by divergence during continuation. They do not identify
the first differing gradient, prove Python/NumPy RNG parity, or establish a
particular faulty CUDA kernel. Different nodes/jobs cannot supply that control.

Two source defects have been established locally:

1. **The reproducibility contract stopped at seeding.** `seed_all` initializes
   Python, NumPy and Torch, including CUDA generators, but the original code never
   required deterministic algorithms, a reproducible cuBLAS workspace or
   deterministic cuDNN selection. `runtime_identity` recorded some default flags;
   it did not enforce them. Equal RNG state therefore did not imply equal CUDA
   gradients. This is the principal runtime hypothesis for the UIT failure.
2. **Optimizer restoration had no semantic parameter-order check.** PyTorch loads
   optimizer state positionally. Exact equality of two anonymous `state_dict`s
   does not prove that each moment belongs to the same named parameter. A new
   regression demonstrates that the original restore silently accepted two
   reversed, same-shaped adapter parameters. The repair rejects this before
   mutating the model. There is no evidence that ordering actually changed in the
   reported UIT jobs; the production factory traverses the same model topology.

The CPU tests do not establish the exact cause of the historical H200 failures.
An authentic same-allocation GPU rerun remains required to validate the runtime
repair. Do not describe this patch or a CPU pass as a completed GPU qualification.

The mechanism is consistent with the version-matched primary references:
[PyTorch 2.9 reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html),
[SDPA kernel selection](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html),
and [optimizer state loading](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.load_state_dict.html).

## Changes and verified invariants

| Files | Behavior |
| --- | --- |
| `src/asr/precision.py`, `src/asr/cli.py` | Set a supported cuBLAS workspace before CUDA initialization; strict deterministic algorithms (`warn_only=False`), cuDNN deterministic on, benchmark off. Apply at CLI entry and allocated-GPU inspection, without reseeding. Missing workspace after CUDA initialization or invalid workspace fails explicitly. Deterministic-kernel failures cannot cause the existing precision probe to fall back from BF16 to FP16. |
| `src/asr/preflight.py` | Bind workspace, deterministic warning mode and cuDNN determinism into runtime identity, alongside existing precision/TF32/driver/hardware flags. |
| `src/asr/model.py` | Make checkpoint recomputation's existing `preserve_rng_state=True` default explicit; retain `use_reentrant=False`. |
| `src/asr/checkpoint.py` | Store optimizer group membership by actual model parameter name; reject mismatches or missing layout before restore. Add a read-only fingerprint of Python, NumPy, Torch CPU and, on CUDA, all CUDA RNG states. |
| `src/asr/training.py` | Optional smoke-only observer sees unscaled accumulated named gradients before clipping and AdamW. It records micro-step occurrence order and RNG fingerprints without drawing randomness. Normal training does not collect these traces. |
| `src/asr/smoke.py` | Same three windows and checkpoint boundary, plus an independent continuous repeat. Compare losses, occurrence order, RNG fingerprints and pre-step named gradients. Verify exact immediate restore of adapter, optimizer, scheduler, scaler and all RNG sources. Preserve final adapter/optimizer/scheduler/scaler/RNG/progress/frozen-base comparisons. Errors identify branch, window and tensor name. Persist `resume-diagnostic.json` with per-step losses, LR, gradient hashes, gradient norms, runtime flags and optimizer ID/name layout. |
| `tests/test_determinism.py`, `tests/test_optimizer_mapping.py` | New regressions for strict RNG-neutral determinism setup, invalid/late workspace refusal and same-shape optimizer reordering. |
| `tests/test_checkpoint.py`, `tests/test_precision.py`, `tests/test_smoke.py` | Check saved parameter-name layout; prohibit deterministic-error precision fallback; execute tiny real Whisper + PEFT at FP32/BF16 with dropout and checkpoint recomputation; verify three-window parity; inject repeat-gradient divergence and require failure before resume; check named mismatch errors. |
| `tests/gpu/test_full_model_smoke.py` | Require the real smoke's additional repeat, exact restore, pre-step parity, unchanged tolerance and deterministic-runtime evidence. |

No changes to `configs/`, `requirements/`, augmentation code, noise assets,
dataset files or the original diagnostic. LoRA q/v topology, rank 8, alpha 16,
dropout 0.05, FP32 parameter storage, BF16 preference, TF32 policy, SDPA dispatch,
gradient checkpointing, AdamW hyperparameters, scheduler, accumulation 16,
10 epochs and augmentation probabilities/SNR remain unchanged. The tolerance
remains `rtol=1e-4`, `atol=1e-5`; immediate restoration and RNG are checked exactly.
Gaussian Noise methodology changes are outside this patch.

New checkpoints require the named optimizer layout. Old smoke checkpoints remain
available for forensic inspection but are intentionally not accepted by the new
restore path without that evidence. Start a NEW smoke run. Code and runtime
identity changes also invalidate previous preflight reports; regenerate both.

## Local validation

Isolated `.venv-cpu`: Python 3.11.16, Torch 2.9.0+cpu, Transformers 4.57.1,
PEFT 0.17.1, NumPy 2.3.4, Accelerate 1.10.1 and huggingface-hub 0.36.0.
The UIT report used Python 3.11.18 and CUDA Torch; these are not identical runtimes.

- Original CPU suite: **129 passed, 7 deselected**.
- Before implementation: new deterministic-policy and reordered-optimizer
  regressions failed (**4 failures**); the reorder case silently restored before
  the repair.
- Final CPU suite: **138 passed, 7 deselected**. The seven existing Linux/GPU/
  snapshot cases require their actual platforms and remain required there.
- Tiny Whisper executes all 44 occurrences in `[16,16,12]` for both continuous
  branches and resumes the last 28 occurrences, including active LoRA dropout,
  non-reentrant checkpoint recomputation, FP32 optimizer moments and FP32/BF16
  autocast profiles. The laptop's oneDNN cannot run BF16 convolution backward;
  this fixture explicitly uses native CPU kernels for all compared branches.
  No test is skipped and GPU backend selection is unaffected by that fixture.
- Expected warnings: legacy TF32 API, Whisper generation-config migration and
  CPU oneDNN/Intel-GPU TF32 capability notices.
- `git diff --check` passes; locked config, requirements and augmentation have no
  diff. No commit or push was performed.

Reproduce on Windows from the checkout:

```powershell
& .\.venv-cpu\Scripts\python.exe -m pytest -q --basetemp=.cache/pytest-review --junitxml=.cache/resume-fix-cpu.xml
git diff --check
git diff -- configs requirements src/asr/augmentation.py
```

Local machine-readable test evidence: `.cache/resume-fix-cpu.xml` (Git-ignored).

## UIT deployment after review and user-managed push

These are operator commands, not actions already executed by this repair.
Use the existing verified dataset, snapshot, noise and Slurm resource paths.
Do not replace the dependency lock or rerun environment setup unnecessarily.

Preserve server changes before updating source. The diagnostic specifically
reported local edits in `tests/test_cli.py`, `tests/test_slurm_scripts.py`,
`tests/test_smoke.py` and untracked `requirements/candidate-resolved-linux.txt`.
The following keeps the stash even after reapplying it. Any conflict stops the
shell; resolve it while preserving both the server guards and new regressions
before proceeding. Do not reset, clean, or discard those files.

```bash
set -euo pipefail
export ASR_PROJECT_ROOT=/datastore/binhvp/BuiTrongTan/ASR
cd "$ASR_PROJECT_ROOT"
git status --short
stash_before=$(git rev-parse -q --verify refs/stash || true)
git stash push -u -m "uit-local-before-resume-fix-$(date +%Y%m%d-%H%M%S)"
stash_after=$(git rev-parse -q --verify refs/stash || true)
git pull --ff-only
if [[ "$stash_after" != "$stash_before" ]]; then
    git stash apply "$stash_after"
fi
git status --short
git diff --check

export ASR_SLURM_ENV="$ASR_PROJECT_ROOT/configs/slurm.local.env"
source "$ASR_SLURM_ENV"
: "${ASR_DATA_ROOT:?Use the existing verified dataset root}"
export ASR_ACCOUNT=uit ASR_PARTITION=normal ASR_QOS=mig2-35
export ASR_GRES=gpu:2g.35gb:1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Keep the already verified ASR_CPUS, ASR_MEM, ASR_TIME and ASR_SIGNAL_SECONDS.
# Smoke now runs 116 training occurrences plus the unchanged max-shape probe;
# ensure the allocation's operational time limit covers the additional repeat.

.venv/bin/python -m pytest tests -m 'not gpu and not snapshot and not linux' -q
.venv/bin/python -m pytest tests/test_slurm_scripts.py tests/test_inference_resume.py tests/test_checkpoint.py -m linux -q
tag=$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d-%H%M%S)
cpu_report="manifests/preflight-cpu-resume-fix-$tag.json"
gpu_report="manifests/preflight-gpu-resume-fix-$tag.json"
smoke_id="smoke-resume-fix-$tag"
printf '%s\n' "CPU=$cpu_report" "GPU=$gpu_report" "SMOKE=$smoke_id"
.venv/bin/python -m asr preflight --config configs/official.yaml \
    --data-root "$ASR_DATA_ROOT" --stage cpu --output "$cpu_report"
bash scripts/submit_job.sh preflight --output "$gpu_report"
```

Wait for the allocated GPU preflight job to complete successfully. Retain the
variables printed above in the same shell (or restore their exact values), then:

```bash
.venv/bin/python - "$cpu_report" "$gpu_report" <<'PY'
import json, sys
for path in sys.argv[1:]:
    report = json.load(open(path))
    assert report['passed'] is True, (path, report['checks'])
PY
bash scripts/submit_job.sh smoke --run-id "$smoke_id" \
    --cpu-report "$cpu_report" --gpu-report "$gpu_report"
```

Wait for that NEW smoke job, then inspect without modifying old runs:

```bash
.venv/bin/python - "$smoke_id" <<'PY'
import json, pathlib, sys
run = pathlib.Path('runs') / sys.argv[1]
report = json.loads((run / 'smoke-report.json').read_text())
assert report['passed'] is True, report['checks']
required = {'training_windows', 'checkpoint_resume', 'gradient_connectivity', 'maximum_shape'}
assert required <= {c['name'] for c in report['checks'] if c['passed'] is True}
c = report['metadata']['comparison']
assert c['window_sizes'] == [16, 16, 12]
assert c['global_update'] == 3 and c['global_micro_step'] == 44
assert c['matched'] and c['continuous_repeat_matched']
assert c['immediate_restore_exact'] and c['pre_step_gradients_matched']
assert (c['rtol'], c['atol']) == (1e-4, 1e-5)
print(json.dumps(c, indent=2))
PY
```

If it fails, preserve `smoke-report.json`, `resume-diagnostic.json` and the new
Slurm `.out/.err`. `continuous_repeat/window-N/...` identifies a runtime repeat
failure before any restore; `immediate_restore/...` identifies load-time
divergence; `resumed/window-N/before_optimizer/...` identifies continuation
divergence. RNG hashes include Python/NumPy and CUDA, and optimizer errors name
the actual adapter parameter rather than only an integer index. A strict-kernel
error is a failed qualification; do not change precision, tolerance or flags to
turn it into a pass. Stop before official training until the authentic GPU
report passes and remaining commissioning gates are separately satisfied.
