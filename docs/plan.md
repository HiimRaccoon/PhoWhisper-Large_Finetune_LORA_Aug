# PhoWhisper-large LoRA ASR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task by task. Use `executing-plans` for inline execution; use delegated execution only if selected by the user. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete locked ASR pipeline, then verify it on UIT and produce the final FP16 model and correctly mapped Train transcripts.

**Architecture:** A Python package exposes a unified CLI over explicit, independently testable stages: audit, asset verification, augmentation, processor/model setup, training/checkpointing, export, and inference. A single-process training loop owns accumulation and progress; immutable, identity-bearing artifacts connect stages. GPU operations run inside Slurm allocations on one H200 MIG instance.

**Tech Stack:** Python 3.11, PyTorch/CUDA, Transformers, PEFT, NumPy, Accelerate where needed, huggingface-hub for explicit asset preparation, pytest, and Slurm. Candidate versions remain those in specification section 15 until verified and locked on UIT.

**Spec:** [specs.md](specs.md), version 1.0. Read [ASR.md](../ASR.md) with the precedence rules in specification section 1.

**Execution tracker:** [tasks.md](tasks.md). It owns task status and evidence links; the checkboxes here describe each task's execution steps. No implementation, installation, remote access, submission, or training has been performed by creating this plan.

## Global Constraints

- Base: `vinai/PhoWhisper-large`, revision `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb`. New PEFT-default adapter from seed 42; no E4 warm start.
- Train: 1,545 sources and expected 5,036 current + user-history occurrences. No assistant utterances, validation, automatic deduplication, external speech training data, or Public/Private Test use.
- Train JSONL SHA-256: `24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8`. Audit mismatches block the official run.
- LoRA: `scope=encoder_decoder`, `rank=8`, `alpha=16`, `dropout=0.05`, `bias=none`, `modules_to_save=null`. Assert 192 exact target modules and 3,932,160 A/B parameters for the verified architecture; only A/B are trainable.
- Training: seed 42, exactly 10 epochs, micro-batch 1, accumulation 16, AdamW, LR `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, linear scheduler, warmup ratio `0.05`, gradient norm limit `1.0`.
- Gradient checkpointing enabled; `use_cache=False` during training. Precision `auto` prefers verified BF16, otherwise FP16 + GradScaler. No QLoRA, NF4, training quantization, or OOM-driven changes to locked settings.
- E4 policy: 50% clean / 50% augmentation branch; speed → volume → noise gates each 0.5; factors `[0.9, 1.0, 1.1]`, gain `[-6, 6]` dB, SNR `[15, 30]` dB. Use specification section 8's SHA-256 RNG and draw order.
- Input: mono signed 16-bit PCM WAV, 16 kHz, `0 < duration <= 30` seconds. Targets use only NFC + trim; no truncation, crop, silent row exclusion, or reference replacement.
- Processor: 80 mel bins, `language=vi`, `task=transcribe`, `predict_timestamps=False`, at most 447 label tokens after removing the leading decoder-start token, padding masked with `-100`.
- Checkpoints every 50 optimizer updates, at epoch boundaries, and after epoch 10. No validation-based selection or early stopping; select the final adapter.
- Export: merged full FP16 model. Inference: original Train audio, offline, greedy, `do_sample=False`, `num_beams=1`, `max_new_tokens=440`, Vietnamese transcription.
- Project: `/datastore/binhvp/BuiTrongTan/ASR`. Dataset: `/datastore/binhvp/BuiTrongTan/data/train+public_test`, read in place. All generated assets, environments, caches, logs, checkpoints, and outputs stay under the project; never access `CS406/`.
- Training is submitted through Slurm and never runs on `login01`. GPU preflight, smoke, export, and inference require compute-node allocations. Cluster account/partition/QoS/GRES/walltime are discovered, not invented.
- Current-run budget, when audit confirms 5,036 occurrences: 315 updates/epoch, 3,150 total updates, 158 warmup updates, and a final accumulation window of 12.
- This plan authorizes no execution by itself. All commands below are future acceptance commands. Do not automatically submit full training before successful preflight and smoke evidence.

## Review Focus

1. A valid reference with the maximum permitted length must retain EOS and produce a valid decoder input; T03 tests the 447/448 boundary and padding with shared PAD/EOS IDs.
2. An apparently safe path may escape through a symlink; T02 and T03 test resolved-path containment for dataset, snapshot/noise references, and outputs without modifying source data.
3. A restart may reconstruct objects that consume RNG or prefetch examples; T07/T09 compare uninterrupted and resumed traversal, augmentation, adapter state, and scheduler state.
4. A preemption signal or concurrent writer may leave incomplete/colliding checkpoints; T07/T12 test incomplete publication, writer exclusion, attempt IDs, and signal forwarding.
5. A model may produce empty text or stop at its token limit; T11 preserves failures and token evidence, rejects incomplete outputs, and never substitutes reference text.

## Implementation Conventions and Review Follow-ups

These choices complete implementation details left open in the review. They do not change locked values or amend the specification. Record them in resolved runtime metadata and verify them before full training.

| Area | Implementation decision and proof |
|---|---|
| Feature shape | Explicit processor max-length padding to 480,000 samples / 3,000 mel frames; features `[1, 80, 3000]`. Validate original duration first; forbid waveform/label truncation. Preserve masks from the processor. T03 verifies shapes. |
| Training dtypes | Load frozen floating base tensors in FP32, keep LoRA A/B and AdamW state in FP32, and use autocast for BF16/FP16 compute. FP16 uses GradScaler; BF16 does not. Record actual tensor/state dtypes and autocast/cache/backend settings. Profile this strategy; stop on OOM rather than silently choosing another one. |
| PEFT and gradients | Generic `PeftModel` with no text-task wrapper (`task_type=None`). Prefer non-reentrant checkpointing. Never place trainable encoder/decoder execution inside `no_grad`. Initial A gradients may be zero because default B is zero; test A connectivity after a nonzero-LR update has changed B. |
| Interpolation | For input length `n` and new length `m=round(n/factor)`, use `numpy.interp(numpy.linspace(0, n-1, m), numpy.arange(n), audio)`, then cast to float32. Factor 1.0 is identity. Transform outputs are float32; power/gain calculations use float64 intermediates. Store golden fixtures under the pinned NumPy stack. |
| Data ordering | Initial implementation uses a private Torch CPU generator with `torch.randperm`, seed `42 + epoch`, and `num_workers=0`. Store/verify permutation hashes. This removes worker-prefetch state from the first resume contract without changing sample order semantics. |
| Resume ordering | Reconstruct and load model, optimizer, scheduler, scaler, sampler, and loader before restoring checkpoint RNG states. Restore RNG last, before the next training example; loader iteration must not consume the training RNG. |
| Merge | Reload the exact base and final adapter in FP32, merge in eval mode, then cast the merged floating weights to FP16. Compare FP16 adapter-reference and merged logits on identical clean inputs with `atol=0.05` and `rtol=0.05` as an initial technical gate; record observed errors. A failed gate requires investigation, not an automatic tolerance increase. |
| Inference | Use `eval()` and `torch.inference_mode()`. Explicitly resolve `temperature=0.0`, no temperature fallback schedule, `return_timestamps=False`, no timestamp output, and the Vietnamese transcription prefix; clear conflicting inherited forced decoder IDs. Assert the resolved behavior remains greedy with the locked 440-token limit. |
| Scheduling | Use an explicit linear-warmup lambda with initial scheduler step 0 and optimizer-before-scheduler ordering. For update index `u=0..3149`, LR multiplier is `u/158` before warmup ends and `(3150-u)/(3150-158)` thereafter; log LR used before each optimizer call. The initial LR is 0, and the scheduler reaches 0 after the final update. |
| Signals | Use an operator-configured Slurm early-warning signal and a Python termination flag. The handler requests a stop; the loop saves only after a completed window. Size the warning lead time from measured window + checkpoint latency. No automatic requeue/submission is assumed. |

## File and Interface Map

All paths below are relative to the project root. Files are planned, not already available.

| Files | Responsibility / owner |
|---|---|
| `pyproject.toml`, `configs/official.yaml`, `src/asr/{__init__,__main__,config,contracts,cli}.py` | Packaging, validated locked configuration, shared artifact records and CLI; T01/T08 |
| `src/asr/{paths,data,audit}.py` | Path boundaries, source projection, PCM decoding, manifests, integrity; T02 |
| `src/asr/{assets,processor}.py` | Snapshot/noise verification and feature/label preparation; T03 |
| `src/asr/augmentation.py` | Pure deterministic waveform transformation and decision logs; T04 |
| `src/asr/{model,precision}.py` | Exact LoRA initialization and compute/storage precision; T05 |
| `src/asr/{training,schedule}.py` | Budget, permutation, accumulation, optimizer updates and loss guards; T06 |
| `src/asr/checkpoint.py` | Atomic publication, state restoration, identity and progress; T07 |
| `src/asr/{artifacts,logging,preflight}.py` | Run/attempt identities, lifecycle, JSONL logs and admission reports; T08 |
| `src/asr/smoke.py` | Isolated full-model resume/accumulation smoke runner; T09 |
| `src/asr/export.py` | Final-checkpoint selection, FP32 merge, FP16 export and reload checks; T10 |
| `src/asr/{inference,outputs}.py` | Offline generation, utterance/source outputs and validation; T11 |
| `scripts/{setup_env,submit_job}.sh`, `scripts/{preflight,smoke,train,export,infer}.slurm`, `configs/slurm.example.env` | Deployment and Slurm lifecycle; T12 |
| `requirements/{candidate,uit-lock}.txt`, `README.md`, `tests/**` | Candidate/exact dependencies, command reference and behavioral verification; T01–T13 |

Shared records live in `contracts.py` with `schema_version=1`: `UtteranceRow`, `AuditResult`, `SnapshotIdentity`, `NoiseManifest`, `AugmentationResult`, `PreparedExample`, `RuntimeProfile`, `TrainingBudget`, `TrainingProgress`, `ArtifactIdentity`, `PreflightReport`, `SmokeReport`, `ExportResult`, and `OutputValidation`. They contain serializable metadata; waveform/tensor fields are excluded from persisted JSON records. `TrainingBundle` lives in `model.py` and holds the model, exact `target_names`, and runtime profile. Define each record's required fields when its owning task first consumes it; later tasks import the same type rather than redefining it.

## Task Execution Protocol

Read the task's specification sections and dependencies first. For T01–T12, use the listed failing behavioral tests, run them to confirm the intended failure, implement the smallest coherent deliverable, then run the same tests to pass. Record command/output evidence in [tasks.md](tasks.md). If execution occurs in a Git repository, commit each verified task using the task's suggested message; do not initialize a repository solely for this plan.

CPU commands below assume the future environment is installed and `asr` is the package console entry point. Model/GPU tests are marked separately and never run on the login node. All tasks begin unchecked.

### T01: Locked configuration and package contract

**Depends on:** none. **Spec:** 1–3, 5, 15.
**Files:** create `pyproject.toml`, `configs/official.yaml`, `requirements/candidate.txt`, `src/asr/{__init__,__main__,config,contracts}.py`, `tests/test_config.py`.
**Interfaces:** `load_config(path: Path) -> OfficialConfig`; `validate_locked(config: OfficialConfig) -> None`. `OfficialConfig` owns model/data/lora/training/augmentation/processor/inference/export settings and operational paths; frozen methodology values cannot be CLI-overridden.

- [ ] Write `test_locked_config_rejects_method_changes` and `test_operational_paths_do_not_change_method_hash`; parameterize locked fields, unknown keys, boolean-as-integer mistakes, and invalid operational paths.
- [ ] Run `pytest tests/test_config.py -q` and confirm failure before implementation.
- [ ] Implement validation, stable canonical JSON/method hash, and explicit operational path defaults from the specification. Copy the official YAML exactly; declare Python 3.11 and candidate stack versions without claiming UIT compatibility.
- [ ] Run the tests and verify these invariants:
  ```python
  cfg = load_config(Path("configs/official.yaml"))
  assert cfg.training.epochs == 10
  assert cfg.lora.scope == "encoder_decoder"
  assert cfg.data.deduplicate is False
  ```
- [ ] Record evidence and, when applicable, commit `feat: define locked ASR configuration`.

### T02: Safe source projection and dataset audit

**Depends on:** T01. **Spec:** 3–4, 16.
**Files:** create `src/asr/{paths,data,audit}.py`; `tests/{test_projection,test_audio_audit,test_paths}.py`.
**Interfaces:** `resolve_beneath(root: Path, relative: str) -> Path`; `project_sources(train_jsonl: Path) -> list[UtteranceRow]`; `read_pcm16(path: Path) -> tuple[np.ndarray, dict]`; `audit_train(config: OfficialConfig, output_dir: Path) -> AuditResult`. `AuditResult` exposes `valid`, counts, source/manifest hashes, manifest path, and error/duplicate/conflict report paths.

- [ ] Write fixtures with current audio, user turns at history indices 0 and 2, assistant-only history, duplicate PCM, conflicting normalized references, missing/cut WAVs, symlink escape, and forbidden environment/state paths.
- [ ] Run `pytest tests/test_projection.py tests/test_audio_audit.py tests/test_paths.py -q`; confirm failure.
- [ ] Implement read-only projection, PCM/WAV hashes, precise payload/format/duration checks, NFC + trim, identity/count assertions, and stable manifest order. Keep duplicate occurrences; any invalid row/conflict blocks the official run.
- [ ] Run those tests; assert:
  ```python
  assert [r.turn_id for r in rows] == ["current", "history_000", "history_002"]
  assert duplicate_audit.utterance_count == 2
  assert conflict_audit.valid is False
  ```
  Verify that reports contain IDs/reasons and exclude labels/context/state; rejected rows are never silently omitted from a trainable manifest.
- [ ] Record evidence and commit, when applicable, `feat: audit Train utterances without altering source data`.

### T03: Verified assets and Whisper processor

**Depends on:** T01–T02. **Spec:** 4.3, 7.1, 7.3, 8.3, 15.
**Files:** create `src/asr/{assets,processor}.py`; `tests/{test_assets,test_processor}.py`.
**Interfaces:** `verify_snapshot(config: OfficialConfig) -> SnapshotIdentity`; `verify_noise(config: OfficialConfig) -> NoiseManifest`; `load_processor(snapshot: SnapshotIdentity) -> WhisperProcessor`; `prepare_example(row: UtteranceRow, audio: np.ndarray, processor: WhisperProcessor) -> PreparedExample`.
`NoiseManifest.entries` is an ordered list of path/hash/license/review/non-speech records; `PreparedExample` contains `input_features`, `labels`, masks and utterance ID.

- [ ] Write asset integrity/containment tests and processor tests for short audio, exactly 30 seconds, 447/448 label tokens, empty normalized text, decoder-start duplication, and PAD/EOS sharing. Use controlled tokenizer fixtures for boundaries plus a marked real-snapshot check.
- [ ] Run `pytest tests/test_assets.py tests/test_processor.py -m "not gpu and not snapshot" -q`; confirm failure.
- [ ] Implement local-only asset validation, file hash inventories, nonempty positive-power licensed/reviewed noise, original transcript retention, and processor max-length padding without content truncation. Resolve `số_noise` and `số_frame_noise` from the specification as the ordered noise count and chosen asset frame count, respectively.
- [ ] Run CPU tests; assert:
  ```python
  assert example.input_features.shape == (1, 80, 3000)
  assert example.labels.shape[-1] <= 447
  assert padded_labels[padding_positions].eq(-100).all()
  ```
  Actual snapshot token-length checks remain required on UIT; a stub test is not evidence that all Train references pass.
- [ ] Record evidence and commit, when applicable, `feat: verify local assets and prepare Whisper inputs`.

### T04: Deterministic E4 waveform views

**Depends on:** T02–T03. **Spec:** 8.
**Files:** create `src/asr/augmentation.py`, `tests/test_augmentation.py`, `tests/fixtures/augmentation-v1.json`.
**Interfaces:** `augmentation_seed(seed: int, epoch: int, utterance_id: str) -> int`; `augment(audio: np.ndarray, row: UtteranceRow, epoch: int, noise: NoiseManifest, seed: int = 42) -> AugmentationResult`. The result returns waveform, branch/gate/parameter decisions and fallback reasons without changing the row/reference.

- [ ] Write a fixed seed assertion and tests that force each gate, factor, gain, noise offset/wrap, zero-power segment, speed overflow, clipping fallback, and repeat/resume call. Use explicit fake RNG draw traces to prove conditional draw order.
- [ ] Run `pytest tests/test_augmentation.py -q`; confirm failure.
- [ ] Implement specification section 8's exact branch/gate order and RNG APIs, the interpolation convention above, gain cap, power/SNR formula, and clean fallback logging. Never redraw parameters or duplicate rows.
- [ ] Run the tests; verify:
  ```python
  assert augmentation_seed(42, 0, "train_0001/current") == 18138155238159038922
  np.testing.assert_array_equal(first.audio, replay.audio)
  assert row.reference == original_reference
  ```
  Golden fixture metadata includes the seed material, pinned NumPy version, noise entry order/hash, and waveform digest; expected values are computed independently of the production function.
- [ ] Record evidence and commit, when applicable, `feat: implement deterministic E4 audio augmentation`.

### T05: Exact LoRA model and precision profile

**Depends on:** T01, T03. **Spec:** 5, 7, 9.1, 14.2.
**Files:** create `src/asr/{model,precision}.py`; `tests/{test_lora_targets,test_precision}.py`.
**Interfaces:** `resolve_lora_targets(model: torch.nn.Module) -> list[str]`; `require_gpu_allocation(env: Mapping[str, str], hostname: str) -> None`; `probe_runtime(config: OfficialConfig, snapshot: SnapshotIdentity) -> RuntimeProfile`; `build_training_model(config: OfficialConfig, snapshot: SnapshotIdentity, runtime: RuntimeProfile) -> TrainingBundle`. `require_gpu_allocation` lives in `precision.py`, rejects login-node execution or missing Slurm allocation, and is reused by later GPU commands.

- [ ] Write topology fixtures checking the exact six module families, exclusion of all other projections, mismatched topology rejection, generic `input_features` forwarding, and BF16-probe/FP16-scaler selection.
- [ ] Run `pytest tests/test_lora_targets.py tests/test_precision.py -m "not gpu" -q`; confirm failure.
- [ ] Implement fresh seed-42 initialization, frozen FP32 base, FP32 A/B, exact-name adapter injection, non-reentrant checkpointing, `use_cache=False`, and the resolved runtime profile. Build no optimizer here. Allocation guards must run before CUDA probing.
- [ ] Run CPU topology tests; on UIT additionally assert:
  ```python
  assert len(bundle.target_names) == 192
  assert sum(p.numel() for p in bundle.model.parameters() if p.requires_grad) == 3_932_160
  assert all("lora_A" in n or "lora_B" in n for n, p in bundle.model.named_parameters() if p.requires_grad)
  ```
  Verify finite encoder/decoder gradients and base freeze without asserting nonzero A gradients on the first backward.
- [ ] Record evidence and commit, when applicable, `feat: initialize exact encoder decoder LoRA scope`.

### T06: Explicit training windows and scheduler

**Depends on:** T01–T05. **Spec:** 9, 11, 16.
**Files:** create `src/asr/{training,schedule}.py`; `tests/{test_accumulation,test_schedule,test_training_guards}.py`.
**Interfaces:** `compute_budget(n: int) -> TrainingBudget`; `epoch_order(n: int, epoch: int, seed: int = 42) -> list[int]`; `create_optimizer(bundle: TrainingBundle, config: OfficialConfig) -> torch.optim.AdamW`; `create_scheduler(optimizer: torch.optim.Optimizer, budget: TrainingBudget) -> torch.optim.lr_scheduler.LambdaLR`; `run_window(bundle, optimizer, scheduler, scaler, examples: Sequence[PreparedExample], progress: TrainingProgress) -> WindowResult`. `WindowResult` is defined here with loss/LR/norm/scaler/success metadata.

`TrainingProgress` records `epoch` (0–9, or 10 at completion), `completed_epochs`, `next_position`, `global_micro_step`, `global_update`, and `permutation_hash`. Successful windows alone advance progress; epoch transition resets `next_position` and installs the next permutation. `TrainingBudget` records `utterances_per_epoch`, `updates_per_epoch`, `total_updates`, and `warmup_updates`.

- [ ] Write behavioral tests against a small differentiable model: average per-utterance loss for windows 16 and 12, exact sample visitation, fixed scale within a window, unscale-before-clip ordering, no advancement on failed steps, and LR boundary values.
- [ ] Run `pytest tests/test_accumulation.py tests/test_schedule.py tests/test_training_guards.py -q`; confirm failure.
- [ ] Implement loss/k normalization, one optimizer/scheduler action per successful window, clipping after accumulation, independent epoch ordering, and hard failure for non-finite loss/gradients, OOM, or scaler skips. Only A/B enter AdamW; do not initialize optimizer state for frozen base tensors.
- [ ] Run those tests; assert:
  ```python
  budget = compute_budget(5036)
  assert (budget.updates_per_epoch, budget.total_updates, budget.warmup_updates) == (315, 3150, 158)
  assert [len(w) for w in windows][-1] == 12
  assert visited_ids == expected_ids
  ```
  For gradient equivalence, average individual model losses; do not use a concatenated token-weighted loss as the reference.
- [ ] Record evidence and commit, when applicable, `feat: train with exact accumulation and fixed update budget`.

### T07: Immutable checkpoints and exact progress restore

**Depends on:** T01, T04–T06. **Spec:** 10–11, 16.
**Files:** create `src/asr/checkpoint.py`; `tests/{test_checkpoint,test_resume}.py`.
**Interfaces:** `save_checkpoint(run_dir: Path, attempt_id: str, state: CheckpointState) -> Path`; `load_checkpoint(path: Path, expected: ArtifactIdentity) -> CheckpointState`; `restore_training(state: CheckpointState, objects: TrainingObjects) -> TrainingProgress`. Define `CheckpointState` and `TrainingObjects` here, including adapter, optimizer/scheduler/scaler, RNG, progress, identities and permutation.

- [ ] Write tests for an interrupted temporary write, existing destination, concurrent writers, mismatch of each protected identity, an epoch-end checkpoint, and a mid-epoch restart with object initialization consuming RNG.
- [ ] Run `pytest tests/test_checkpoint.py tests/test_resume.py -q`; confirm failure.
- [ ] Implement same-filesystem temporary publication, file hashes/completion marker, atomic rename and exclusive attempt ownership. Use unique attempt UUIDs independent of Slurm job ID. Persist after completed windows with no pending gradients; restore RNG last.
- [ ] Run those tests and compare uninterrupted/resumed sample order, augmentation digests, LR trace, adapter and optimizer state. Check:
  ```python
  assert resumed.next_position == uninterrupted.next_position
  assert resumed.global_update == uninterrupted.global_update
  assert resumed.permutation_hash == uninterrupted.permutation_hash
  ```
  Repeat around a partial window and the epoch transition; unreadable/incomplete checkpoints never enter latest-checkpoint discovery.
- [ ] Record evidence and commit, when applicable, `feat: preserve complete training state across resume`.

### T08: Run lifecycle, logs, admission reports, and CLI

**Depends on:** T01–T07. **Spec:** 6, 11, 14.1, 14.4, 16.
**Files:** create `src/asr/{artifacts,logging,preflight,cli}.py`; `tests/{test_artifacts,test_preflight,test_cli,test_run_loop}.py`; extend `src/asr/__main__.py` and `src/asr/training.py`.
**Interfaces:** `create_run(config: OfficialConfig, identity: ArtifactIdentity, run_id: str) -> Path`; `start_attempt(run_dir: Path) -> str`; `run_preflight(config: OfficialConfig, stage: Literal["cpu", "gpu"], output: Path) -> PreflightReport`; `check_admission(config: OfficialConfig, preflight: Sequence[PreflightReport], smoke: SmokeReport) -> None`; `execute_run(config: OfficialConfig, run_dir: Path, resume_from: Path | None = None) -> TrainingProgress` in `training.py`; `main(argv: Sequence[str] | None = None) -> int`.

- [ ] Write tests for duplicate run IDs, malformed/tampered reports, stale input hashes, missing smoke, forbidden lifecycle transitions, complete JSONL log fields, CLI nonzero error exits, all 10 epoch transitions, and checkpoint triggers at update 50 and epoch/final boundaries.
- [ ] Run `pytest tests/test_artifacts.py tests/test_preflight.py tests/test_cli.py tests/test_run_loop.py -q`; confirm failure.
- [ ] Implement canonical identities, run/attempt directories, structured logs and lifecycle states. Admission binds method/runtime/config, dataset/utterance/noise/snapshot identities and candidate-versus-locked dependency metadata; paths/attempt names alone do not invalidate equivalent inputs. CPU preflight checks references/token lengths; GPU preflight uses T05's allocation guard and real runtime probes. `execute_run` connects projection/views/processor/windows to checkpointing and logs, honoring the resume cursor and stop flag; checkpoint events sharing one boundary publish once with all reasons recorded, and the final adapter is written only at terminal progress.
- [ ] Run those tests; verify `train` cannot call model training without valid CPU/GPU preflight and matching smoke. Register `audit`, `preflight`, `train` and `resume`; unsupported later commands must fail clearly until their owning tasks implement them. `train` always initializes a fresh official adapter; `resume` restores the same run with a new attempt.
- [ ] Record evidence and commit, when applicable, `feat: gate ASR runs with identity bound preflight reports`.

### T09: Full-model smoke and continuation verification

**Depends on:** T01–T08. **Spec:** 14.2–14.3.
**Files:** create `src/asr/smoke.py`, `tests/test_smoke.py`, `tests/gpu/test_full_model_smoke.py`; extend `src/asr/cli.py`.
**Interfaces:** `run_smoke(config: OfficialConfig, audit: AuditResult, reports: Sequence[PreflightReport], run_id: str) -> SmokeReport`. `SmokeReport` exposes fixture IDs, protected identities, profile, checks/tolerances, artifact hashes and `passed`.

- [ ] Write CPU harness tests rejecting warm-start use and fake GPU success; write a separately marked real-model smoke test comparing uninterrupted and resumed execution from the same initial adapter.
- [ ] Run `pytest tests/test_smoke.py -q`; confirm failure. GPU tests stay pending until T13; skipped tests are not passes.
- [ ] Implement an isolated `smoke-` run with 44 manifest occurrences in windows 16, 16, 12, using the official LR schedule/budget with a smoke-only execution limit. Save/resume after the first window; this does not change the official config or its epoch budget. Include a maximum-shape 30-second/447-label technical probe, allocation checks and memory measurements.
- [ ] Run CPU tests and implement the separately marked GPU acceptance check: sample/augmentation/LR/progress equality, base-weight equality, correct dtype/scaler behavior and adapter-state comparison at `rtol=1e-4` / `atol=1e-5` in the same environment. Verify A gradient connectivity after B has changed at a nonzero LR. Actual execution of `pytest tests/gpu/test_full_model_smoke.py -m gpu -q` occurs in T13 inside an allocation; failed comparisons block admission and must not trigger automatic tolerance adjustment.
- [ ] Complete this software task with passing CPU evidence and the real-model test/harness ready to run; leave actual GPU execution/report acceptance pending T13/G3. Never synthesize a passing `SmokeReport`. Commit code when applicable, `feat: verify full model training checkpoint and resume`.

### T10: Verified final adapter merge and FP16 export

**Depends on:** T03, T05, T07–T08. **Spec:** 12, 17.
**Files:** create `src/asr/export.py`, `tests/test_export.py`, `tests/gpu/test_export_reload.py`; extend `src/asr/cli.py`.
**Interfaces:** `export_final(config: OfficialConfig, checkpoint: Path, output_dir: Path) -> ExportResult`. `ExportResult` exposes identities, file hashes, dtype inventory, numerical checks, reload result and `valid`.

- [ ] Write tests rejecting non-final/epoch-9 checkpoints, base/adapter identity mismatch, existing destinations, leftover adapter wrappers, non-finite weights and incomplete local exports.
- [ ] Run `pytest tests/test_export.py -m "not gpu" -q`; confirm failure.
- [ ] Implement final-progress assertions, exact-base FP32 reload + FP32 adapter merge in eval mode, then FP16 conversion and atomic publication. Include processor/tokenizer/generation config and provenance. Launch a fresh-process offline reload with network downloads disabled.
- [ ] Run CPU tests on a tiny Whisper fixture and UIT GPU checks on the actual final model when available. Verify floating weights are FP16, logits pass the declared `0.05/0.05` tolerance and local generation works without PEFT. An untrained tiny fixture proves mechanics only; it cannot certify the final export.
- [ ] Record evidence and commit, when applicable, `feat: export the final merged FP16 Whisper model`.

### T11: Offline Train transcription and completeness validation

**Depends on:** T02–T03, T08, T10. **Spec:** 13–14, 17.
**Files:** create `src/asr/{inference,outputs}.py`; `tests/{test_inference,test_outputs}.py`; extend `src/asr/cli.py`.
**Interfaces:** `transcribe_train(config: OfficialConfig, manifest: Path, exported: ExportResult, output_dir: Path) -> OutputValidation`; `group_transcripts(rows: Sequence[UtteranceRow], predictions: Sequence[dict]) -> list[dict]`; `validate_outputs(manifest: Path, utterance_file: Path, source_file: Path, exported: ExportResult) -> OutputValidation`.

- [ ] Write tests for interleaved history, sources without history, repeated/missing/extra IDs, wrong audio/hash/model, empty text, EOS absence at the 440-token limit, and inherited sampling/timestamp/prefix settings.
- [ ] Run `pytest tests/test_inference.py tests/test_outputs.py -q`; confirm failure.
- [ ] Implement local-only eval/inference-mode generation on clean original audio, explicit greedy generation resolution and raw token evidence. Assemble source records in original JSONL order with original history indices. No ground-truth fallback, reference fields, assistant rows or Public Test input enter transcript artifacts.
- [ ] Run tests; verify exactly one source record per source, exact utterance coverage, and zero hidden failures. Write `output-validation.json` even on failure; publish a completed transcript set only when checks pass. Register `infer-train` and `validate-output`.
- [ ] Record evidence and commit, when applicable, `feat: generate and validate mapped Train transcripts`.

### T12: UIT environment setup and Slurm lifecycle

**Depends on:** T01, T05, T07–T11. **Spec:** 3, 10, 14–16.
**Files:** create `scripts/{setup_env,submit_job}.sh`, `scripts/{preflight,smoke,train,export,infer}.slurm`, `configs/slurm.example.env`, `tests/test_slurm_scripts.py`; create/extend `README.md`.
**Interfaces:** `submit_job.sh <preflight|smoke|train|resume|export|infer> [CLI arguments]` validates resource settings, then invokes `sbatch` with explicit command-line resource/output flags. Slurm scripts invoke the shared CLI; `resume` uses `train.slurm` with a checkpoint argument.

- [ ] Write tests using fake `sbatch`/`srun` executables for missing account/partition/GRES, paths containing spaces, argument forwarding, log directories, exit status propagation, signals and a restarted job with the same Slurm job ID.
- [ ] Run `pytest tests/test_slurm_scripts.py -q`; confirm failure.
- [ ] Implement one task/process/GPU instance, project-local venv/caches/temp/logs, and explicit project paths. Read Slurm resources from operational settings and pass them to `sbatch`; do not interpolate shell variables inside `#SBATCH` directives. Use `srun` for the Python job step and verify the configured warning signal reaches the training process. Create log parents before submission.
- [ ] Run script tests and `for script in scripts/*.sh scripts/*.slurm; do bash -n "$script" || exit; done` on Ubuntu so every file is parsed. Document the warning signal, measured-lead-time requirement, allocation guards, CPU versus GPU commands, setup, explicit snapshot preparation, noise review, train, resume, export and inference. Scripts must never auto-submit official training after another job.
- [ ] Record evidence and commit, when applicable, `feat: run the ASR lifecycle through configurable Slurm jobs`.

### T13: UIT commissioning and preflight evidence

**Depends on:** T01–T12. **Spec:** 3–5, 7–9, 14–16.
**Files:** create `requirements/uit-lock.txt` only after actual environment verification; update `README.md` and operational `configs/slurm.example.env` without changing the official method. Reports go under `manifests/` and isolated `runs/smoke-<id>/`.
**Interfaces:** use the CLI and scripts from T08–T12. Produce identity-bound CPU/GPU `PreflightReport` records and a passing `SmokeReport` for the actual server inputs/environment.

- [ ] Inspect UIT paths and allowed cluster resources read-only; confirm the Train identity and GPU profile. Record account/partition/QoS/GRES/time limits, filesystem behavior, GPU visibility and driver/CUDA metadata without accessing `CS406/`.
- [ ] Prepare the isolated Python 3.11 environment from candidate requirements, the exact local pretrained snapshot and independently licensed/reviewed non-speech noise. Runtime still has no automatic downloads; do not fabricate licensing/review evidence. Produce the exact dependency lock after compatibility checks.
- [ ] Run the CPU gate:
  ```bash
  asr audit --config configs/official.yaml --data-root "$DATA_ROOT" --output manifests/train
  asr preflight --config configs/official.yaml --data-root "$DATA_ROOT" --stage cpu --output manifests/preflight-cpu.json
  ```
  Operational paths resolve from the same configuration and `DATA_ROOT`; expected audit is 1,545/5,036, exact Train hash, no unresolved errors and all label lengths <=447.
- [ ] Submit only GPU preflight and smoke through the wrapper, with official inputs and the isolated smoke run ID. Measure maximum-shape memory, window duration and checkpoint duration; set and test the early-warning lead time. Repeat relevant checks after any runtime/asset change.
- [ ] Verify full-model target/count assertions, BF16 or FP16 profile, gradient connectivity, resumed-state comparison and stale-report rejection. No skipped check can establish admission. Save command/output/report hashes and update [tasks.md](tasks.md).
- [ ] Declare `READY_FOR_OFFICIAL_RUN` only when both preflight reports and smoke pass for matching identities. Otherwise list the concrete blocker and corrective preparation; do not submit T14 or alter locked methodology.

### T14: Official run and end-to-end acceptance

**Depends on:** T01–T13 and an operator-scheduled official execution. **Spec:** 9–13, 17.
**Files/artifacts:** no new training implementation; create `runs/<run_id>/**` and update execution evidence in [tasks.md](tasks.md). Source dataset and locked configuration remain unchanged.
**Interfaces:** `train`, `resume`, `merge-export`, `infer-train` and `validate-output` from earlier tasks. Consume matching T13 reports and smoke evidence; produce the final adapter, full FP16 export, 5,036 utterance predictions and 1,545 source transcript records.

- [ ] Recompute/protect official identities, reserve a unique official run ID and verify matching admission reports. Initialize a new seed-42 adapter from pretrained base; never continue from smoke weights.
- [ ] Submit the official training job through `submit_job.sh train` at the scheduled operation stage. Preserve each allocation's attempt/log metadata; on interruption use `submit_job.sh resume` with the latest complete compatible checkpoint.
- [ ] Verify exactly 10 completed epochs, 315 successful updates per epoch and 3,150 total for the confirmed dataset. Check 50-update/epoch/final checkpoints and one visit per occurrence per epoch; any failed/scaler-skipped update blocks completion.
- [ ] Run `merge-export` in a compute allocation using the final epoch-10 checkpoint only; verify actual full-model offline reload, finite FP16 weights, numerical comparison and artifact hashes.
- [ ] Run `infer-train` and `validate-output` on original audio through the GPU scripts. Require all 5,036 utterances and 1,545 source records, correct IDs/indices/hashes and no unresolved empty/truncated/error output.
- [ ] Finalize `run-manifest.json`, `run-summary.json` and `output-validation.json` with acceptance evidence. Mark only the stages actually achieved; preserve all prior checkpoints/attempts. An interrupted/blocked run remains incomplete.

## Dependency Order and Verification Gates

Execute T01 → T02 → T03 → T04/T05 → T06 → T07 → T08 → T09/T10 → T11 → T12 → T13 → T14. The slashes indicate independent dependencies, not a request to spawn agents. Sequential execution is valid throughout.

| Gate | Required evidence | Unlocks |
|---|---|---|
| G1: CPU implementation | T01–T12 CPU tests pass; script syntax checks pass on Ubuntu; CLI help and locked-config rejection work | UIT commissioning |
| G2: Input identity | Server Train hash/counts, all valid audio/labels, verified snapshot/noise manifests and dependency lock | GPU preflight |
| G3: GPU readiness | Actual allocation/profile, target/trainable assertions, dtype/kernel/memory probes, passing uninterrupted/resumed smoke, tested signal forwarding | Scheduling the official run |
| G4: Training completion | Fresh official initialization, exactly 10 epochs / 3,150 updates, immutable final checkpoint | Final export |
| G5: Model export | Correct base/final adapter, valid merged FP16 artifact, fresh-process offline reload and numerical report | Official Train inference |
| G6: End-to-end output | Complete mapping/coverage, output validation and run summary with hashes | System acceptance |

G1's aggregate CPU verification command is `pytest tests -m "not gpu and not snapshot" -q`, followed by the per-file Bash syntax loop in T12. Register both markers in `pyproject.toml`; excluded tests remain pending, not passed. T01–T12 completion means their software and CPU-verifiable deliverables are ready. The actual snapshot/GPU tests and final-model checks belong to T13/G3 and T14/G5 respectively, so they do not create a circular dependency on commissioning or training.

## Specification Coverage

| Specification section | Implementation tasks |
|---|---|
| 1. Objectives and requirement precedence | T01, T08, T14 |
| 2. System scope | T01–T14 |
| 3. Environment and filesystem | T01–T03, T05, T08, T12–T14 |
| 4. Dataset identity and contracts | T02–T03, T11, T13 |
| 5. Final locked configuration | T01, T05–T06, T08 |
| 6. Architecture and artifact flow | T01–T12 |
| 7. Snapshot, LoRA, processor | T03, T05, T13 |
| 8. Deterministic augmentation | T03–T04, T07, T09 |
| 9. Training and budget | T05–T06, T09, T14 |
| 10. Checkpointing and resume | T07, T09, T12, T14 |
| 11. Logging and artifact management | T06–T08, T14 |
| 12. Merge and FP16 export | T10, T14 |
| 13. Offline inference and outputs | T11, T14 |
| 14. Preflight, tests, CLI, Slurm | T01–T13 |
| 15. Dependencies and reproducibility | T01, T03–T09, T12–T13 |
| 16. Failure policy and blockers | T01–T14 |
| 17. Acceptance criteria | T13–T14, G1–G6 |

## Handoff and Completion Rules

Creating these documents completes planning only. Implementation begins in a later execution stage; no dependency installation, dataset modification, full-training submission or remote work is implied by this documentation change. T13 cannot be marked complete from local/stub tests, and T14 cannot be marked complete from a passing smoke test.

Review this plan and its conventions before implementation. Inline execution uses `executing-plans`; delegated execution uses `subagent-driven-development` only when selected. The tracker records evidence and blockers, and the specification remains authoritative if an implementation detail conflicts with a locked requirement.

