# UIT ASR GPU Smoke / Checkpoint Resume Failure — Diagnostic Evidence

**Prepared:** 2026-11-07 (Asia/Ho_Chi_Minh)  
**Project:** `HiimRaccoon/PhoWhisper-Large_Finetune_LORA_Aug`  
**Observed source revision on UIT server:** `db2d7205e5e5578f10d0cc4d8e214756c4da8547` (`main`)  
**Server project:** `/datastore/binhvp/BuiTrongTan/ASR`  
**Issue status:** Reproduced on two H200 nodes. **Root cause not yet established.**

## Executive summary

The pinned PhoWhisper-large model and LoRA passed CPU and allocated H200 MIG GPU preflight. A 44-utterance GPU smoke test that compares uninterrupted training against checkpoint-resumed training **failed**. Subsequent diagnostics verified **exact** restoration of the LoRA adapter, optimizer `state_dict`, scheduler `state_dict`, and Torch/CUDA RNG **immediately after load**. Nevertheless, after subsequent training windows, an AdamW first-moment tensor (`exp_avg`) differs beyond the strict smoke-test tolerance. This narrows the investigation to behavior after restore (gradient computation, dropout/CUDA reproducibility, parameter-to-optimizer-state mapping, or other runtime differences); it does **not** prove CUDA non-determinism or a faulty checkpoint implementation.

**Important:** Do not start the official 10-epoch fine-tuning until an authentic GPU smoke pass with the same methodology and a validated resume path. Do not simply loosen comparison tolerance, disable checks, change optimizer/hyperparameters, or bypass checkpoint validation.

## Environment and official design (unchanged)

- Slurm account `uit`, partition `normal`, QoS `mig2-35`, GRES `gpu:2g.35gb:1`.
- Node(s) observed: `h200node01` (11292), `h200node02` (11323).
- GPU: NVIDIA H200 MIG 2g.35gb, CUDA 12.8, driver 580.126.20.
- PyTorch 2.9.0+cu128, Python 3.11.18, Transformers 4.57.1, PEFT 0.17.1.
- GPU runtime compute dtype `bfloat16`; `scaler_enabled = false`.
- Model: `vinai/PhoWhisper-large`, pinned HF snapshot revision `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb`.
- LoRA: encoder and decoder q/v, rank 8, alpha 16, dropout 0.05; 192 modules, 3,932,160 trainable parameters.
- Official training design: 5,036 Train utterances, 10 epochs, batch 1, gradient accumulation 16, AdamW LR 1e-4, no validation/dedup. Smoke test uses 44 examples in windows `[16, 16, 12]`, including optimizer steps, full checkpoint, resume parity, and max-shape probe.
- Gaussian white noise was generated on the server (seed 42, 600 s, 16 kHz, PCM16) and passed *existing technical* noise-manifest validation. The repository's older written methodology still requires real ambient recordings and **must later be updated as a separate, explicitly approved change**. Do not present synthetic Gaussian as externally recorded audio. Noise waveform SHA256: `120f10fee8f193a6b03bf94451fb4ac7d91fa2d13308c5a43e79fb93376a6cec`; manifest SHA256: `5215401662e58086b7e69a76d5bd504a5275d118246cb2dc2623732a7c0fbe30`.

## Preflight — successful

**CPU preflight:** `manifests/preflight-cpu.json` => `passed: true`; checks `dataset`, `snapshot`, `noise`, `dependencies`, `labels` all PASS.

**GPU preflight Slurm job 11123:** COMPLETED (`0:0`) in 00:01:17; `manifests/preflight-gpu.json` => `passed: true`; checks `allocation`, `dataset`, `snapshot`, `noise`, `dependencies`, `labels`, `gpu_probe` all PASS. GPU probe: 192 target modules; 3,932,160 trainable parameters; 7,110,786,560 peak allocated bytes. Nonfatal warnings about `torch_dtype` and legacy TF32 API appear in stderr.

## Failure 1 — official smoke, Slurm job 11226

- Job: `asr-smoke`, **FAILED** (`1:0`), 00:02:01.
- Report: `runs/smoke-commission-001/smoke-report.json` => `passed: false`.
- Failure check: `execution_error`.
- Message:

```text
AssertionError: Tensor-likes are not close!
Mismatched elements: 1891 / 10240 (18.5%)
Greatest absolute difference: 0.0002201143215643242 at index (802, 5) (up to 1e-05 allowed)
Greatest relative difference: 788.9552612304688 at index (598, 2) (up to 0.0001 allowed)
```

- Complete first-window checkpoint exists at:
  `runs/smoke-commission-001/checkpoints/676aef251b3c41d2853c77341676e9b7-baseline/update-000001-smoke`.
- Checkpoint metadata: `global_update=1`, `global_micro_step=16`; `adapter.safetensors`, `state.pt`, `COMPLETE.json` exist.

## Failure 2 — diagnostic 002, Slurm job 11292, h200node01

- Job: `asr-smoke-diag`, **FAILED** (`1:0`), 00:01:40.
- Report: `runs/smoke-diag-002/smoke-report.json` => `passed: false`.
- Diagnostic ran with a wrapper script under `.cache/t13-logs/`, not a change to project source.

```text
Adapter exact restore: True
Torch/CUDA RNG exact restore: True
FIRST TENSOR MISMATCH
Path: state/state/1/exp_avg
Shape: (1280, 8)
Dtype: torch.float32
Mismatched elements: 988 / 10240 (9.6%)
Max absolute difference: 0.00022666560835205019
```

- Exact comparison diagnostic identifies `exp_avg` as the first failing tensor in the optimizer state; the adapter comparison preceding this optimizer-state comparison did not raise.

## Failure 3 — diagnostic 003, Slurm job 11323, h200node02

- Job: `asr-smoke-diag`, **FAILED** (`1:0`), 00:01:40.
- Report: `runs/smoke-diag-003/smoke-report.json` => `passed: false`.

```text
Optimizer exact restore: True
Scheduler exact restore: True
Adapter exact restore: True
Torch/CUDA RNG exact restore: True
FIRST TENSOR MISMATCH
Path: state/state/1/exp_avg
Shape: (1280, 8)
Dtype: torch.float32
Mismatched elements: 9966 / 10240 (97.3%)
Greatest absolute difference: 0.009125277400016785 at index (555, 3) (up to 1e-05 allowed)
Greatest relative difference: 13792.015625 at index (109, 2) (up to 0.0001 allowed)
```

Note: Numeric mismatch magnitude differs between 11292 and 11323; these are different Slurm jobs and different compute nodes, **not a controlled same-node A/B comparison**. Do not assume node choice or BF16 is the cause based only on these observations.

## Source pathways to inspect

1. `src/asr/smoke.py`:
   - `compare_resume` creates original objects with `factory()`, runs windows `[16,16,12]`, checkpoints after window 1, captures expected state after window 3, deletes original, creates a fresh `factory()`, `load_checkpoint`, `restore_training`, runs windows 2 and 3, and compares adapter, optimizer, scheduler, scaler, RNG, progress, frozen base.
   - `_compare` enforces `torch.testing.assert_close(..., rtol=1e-4, atol=1e-5)`.
   - `run_smoke` selects first 44 examples from deterministic epoch 0 ordering and calls `augment` / `prepare_example` once per row before branch comparison.
2. `src/asr/checkpoint.py`: `capture_training`, `capture_rng`, `save_checkpoint`, `load_checkpoint`, `restore_training`, `restore_rng`; check optimizer state association with parameter identity and all RNG sources.
3. `src/asr/model.py`: `initialize_adapter` reseeds Python, NumPy and Torch, injects LoRA, enables gradient checkpointing (`use_reentrant=False`), moves model to CUDA.
4. `src/asr/training.py`: `run_window` calls `bundle.model.train()` and forward/backward for each example; gradients accumulate, then clip; `optimizer.step()`, `scheduler.step()`; progress advances.
5. `src/asr/precision.py`: autocast / TF32 / CuDNN controls; inspect any untracked nondeterministic pathways.
6. `tests/test_smoke.py`, `tests/test_checkpoint.py`, `tests/test_inference_resume.py`, and GPU regression tests.

## Investigation plan for Codex

1. **Confirm source tree** against `db2d720...`; preserve unrelated local working changes.
2. Add temporary diagnostics (not changes to locked official methodology) to map AdamW state index 1 to its **parameter name**, and compare before/after each step, preferably by named parameter rather than integer optimizer key.
3. At the moment of resume, verify adapter, optimizer, scheduler and **all** relevant RNG state (Python, NumPy, CPU Torch and CUDA). Existing diagnostics verified adapter, optimizer, scheduler and Torch/CUDA; they do not by themselves prove Python/NumPy RNG parity.
4. Run a controlled deterministic comparison **on a single allocated H200 MIG GPU job**: (a) continuous vs independently repeated continuous run, and (b) continuous vs checkpoint/resumed run; compare gradients **before the next optimizer step**, not only final AdamW moments. Record per-step losses, learning rates, gradient norms, RNG checksums and tensor names. Treat sample/augmentation order and model train/eval state as explicit invariants.
5. Check CUDA/BF16 reproducibility, dropout and gradient-checkpointing behavior; any deterministic settings must be implemented coherently before CUDA initialization if required, and should not silently alter the official numerical-method assumptions. Distinguish *ordinary numerical reproducibility limits* from a material resume bug.
6. If a legitimate bug is found, make the **minimum justified** source fix and regression tests; do not merely increase tolerance. Ensure checkpoint round-trip identities, correct optimizer parameter mapping and resumed optimizer steps are validated.
7. Run CPU unit/integration checks, then submit a **new** GPU smoke run ID; keep all previous failed jobs/reports. Require `training_windows`, `checkpoint_resume`, `gradient_connectivity`, `maximum_shape` all PASS, `window_sizes=[16,16,12]`, `global_update=3`, and an authentic `matched=true`. Perform remaining commissioning (real signal handling, snapshot forward test) separately.

## Constraints and local state

- Do not modify `CS406/` or dataset Train files; do not add validation data, change LoRA scope/size, epochs, LR, accumulation, augmentation probabilities or SNR range in this bugfix.
- Do not start official training while GPU smoke fails.
- On server, `git status --short` showed local modifications to **three test files** (`tests/test_cli.py`, `tests/test_slurm_scripts.py`, `tests/test_smoke.py`) made for Linux `HOSTNAME=login01` guards, plus untracked `requirements/candidate-resolved-linux.txt`. Avoid destructive `git reset --hard` or clobbering server-local changes when deploying a fix.
- The user explicitly wants **source fixes performed by Codex on the local project**, reviewed/pushed, then pulled and verified on UIT server. Diagnostic scripts in `.cache` and logs in `runs` are transient and should not be mistaken for official source modifications.

## Evidence locations on UIT server

```text
manifests/preflight-cpu.json
manifests/preflight-gpu.json
runs/smoke-commission-001/smoke-report.json
runs/smoke-diag-002/smoke-report.json
runs/smoke-diag-003/smoke-report.json
runs/slurm/asr-smoke-11226.out
runs/slurm/asr-smoke-11226.err
runs/slurm/asr-smoke-diag-11292.out
runs/slurm/asr-smoke-diag-11292.err
runs/slurm/asr-smoke-diag-11323.out
runs/slurm/asr-smoke-diag-11323.err
.cache/t13-logs/diagnose_smoke.py
.cache/t13-logs/diagnose_smoke_003.py
```

This report is assembled from operator-provided UIT logs and inspected repository source; it is not a claim that the author personally executed these jobs or established a root cause.
