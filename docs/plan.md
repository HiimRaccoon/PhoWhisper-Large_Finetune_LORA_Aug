# PhoWhisper-large LoRA ASR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task by task. Use `executing-plans` for inline execution; use delegated execution only if selected by the user. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete locked ASR pipeline, then verify it on UIT and produce the final FP16 model and correctly mapped Train transcripts.

**Architecture:** A Python package exposes a unified CLI over explicit, independently testable stages: audit, asset verification, augmentation, processor/model setup, training/checkpointing, export, and inference. A single-process training loop owns accumulation and progress; immutable, identity-bearing artifacts connect stages. GPU operations run inside Slurm allocations on one H200 MIG instance.

**Tech Stack:** Python 3.11, PyTorch/CUDA, Transformers, PEFT, NumPy, Accelerate where needed, huggingface-hub for explicit asset preparation, pytest, and Slurm. Candidate versions remain those in specification section 15 until verified and locked on UIT.

**Spec:** [specs.md](specs.md), version 1.0. Read [ASR.md](../ASR.md) with the precedence rules in specification section 1.

**Execution tracker:** [tasks.md](tasks.md). It owns task status and evidence links; the checkboxes here describe each task's execution steps. No ASR implementation, dependency installation, remote access, submission, or training has been performed by this planning revision. Existing Git publication is not implementation or commissioning evidence.

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
| Feature shape | Explicit processor max-length padding to 480,000 samples / 3,000 mel frames; features `[1, 80, 3000]`. Validate original duration first; forbid waveform/label truncation. Pass explicit `return_attention_mask=True` and retain the processor's frame mask; the detailed shape/valid-frame contract and compatibility tests are in T03. |
| Training dtypes | Load frozen floating base tensors in FP32, keep LoRA A/B and AdamW state in FP32, and use autocast for BF16/FP16 compute. FP16 uses GradScaler; BF16 does not. Record actual tensor/state dtypes and autocast/cache/backend settings. Profile this strategy; stop on OOM rather than silently choosing another one. |
| PEFT and gradients | Generic `PeftModel` with no text-task wrapper (`task_type=None`). Prefer non-reentrant checkpointing. Never place trainable encoder/decoder execution inside `no_grad`. Initial A gradients may be zero because default B is zero; test A connectivity after a nonzero-LR update has changed B. |
| Interpolation | For input length `n` and new length `m=round(n/factor)`, use `numpy.interp(numpy.linspace(0, n-1, m), numpy.arange(n), audio)`, then cast to float32. Factor 1.0 is identity. Transform outputs are float32; power/gain calculations use float64 intermediates. Store golden fixtures under the pinned NumPy stack. |
| Data ordering | Initial implementation uses a private Torch CPU generator with `torch.randperm`, seed `42 + epoch`, and `num_workers=0`. Store/verify permutation hashes. This removes worker-prefetch state from the first resume contract without changing sample order semantics. |
| Resume ordering | Reconstruct and load model, optimizer, scheduler, scaler, sampler, and loader before restoring checkpoint RNG states. Restore RNG last, before the next training example; loader iteration must not consume the training RNG. |
| Merge | Reload the exact base and final adapter in FP32, merge in eval mode, then cast the merged floating weights to FP16. Compare FP16 adapter-reference and merged logits on identical clean inputs with `atol=0.05` and `rtol=0.05` as an initial technical gate; record observed errors. A failed gate requires investigation, not an automatic tolerance increase. |
| Inference | Use `eval()` and `torch.inference_mode()`. Explicitly resolve `temperature=0.0`, no temperature fallback schedule, `return_timestamps=False`, no timestamp output, and the Vietnamese transcription prefix; clear conflicting inherited forced decoder IDs. Assert the resolved behavior remains greedy with the locked 440-token limit. |
| Scheduling | Use an explicit linear-warmup lambda with initial scheduler step 0 and optimizer-before-scheduler ordering. For update index `u=0..3149`, LR multiplier is `u/158` before warmup ends and `(3150-u)/(3150-158)` thereafter; log LR used before each optimizer call. The initial LR is 0, and the scheduler reaches 0 after the final update. T06 defines successful zero-LR calls, exact boundary indices and restored scheduler state. |
| Inference continuation | T11 keeps a durable ordered prefix in a private JSONL journal, bound to model/config/manifest/resolved-generation hashes. Restart validates the prefix and resumes at its next index; final publication requires complete output validation. Public transcript schemas and generation parameters remain unchanged. |
| Signals | Use an operator-configured Slurm early-warning signal and a Python termination flag. The handler requests a stop; the loop saves only after a completed window. Size the warning lead time from measured window + checkpoint latency. No automatic requeue/submission is assumed. |

## File and Interface Map

All paths below are relative to the project root. Files are planned, not already available.

| Files | Responsibility / owner |
|---|---|
| `.gitignore`, `pyproject.toml`, `configs/official.yaml`, `src/asr/{__init__,__main__,config,contracts,cli}.py` | Packaging, validated locked configuration, shared artifact records and CLI; T01/T08 |
| `src/asr/{paths,data,audit}.py` | Path boundaries, source projection, PCM decoding, manifests, integrity; T02 |
| `src/asr/{assets,processor}.py` | Snapshot/noise verification and feature/label preparation, including explicit frame masks; T03 |
| `src/asr/augmentation.py` | Pure deterministic waveform transformation and decision logs; T04 |
| `src/asr/{model,precision}.py` | Exact LoRA initialization and compute/storage precision; T05 |
| `src/asr/{training,schedule}.py` | Budget, permutation, accumulation, optimizer updates and loss guards; T06 |
| `src/asr/checkpoint.py` | Atomic publication, state restoration, identity and progress; T07 |
| `src/asr/{artifacts,logging,preflight}.py` | Run/attempt identities, lifecycle, JSONL logs and admission reports; T08 |
| `src/asr/smoke.py` | Isolated full-model resume/accumulation smoke runner; T09 |
| `src/asr/export.py` | Final-checkpoint selection, FP32 merge, FP16 export and reload checks; T10 |
| `src/asr/{inference,outputs}.py` | Offline generation, durable continuation journal, utterance/source grouping and validation; T11 |
| `scripts/{setup_env,submit_job}.sh`, `scripts/{preflight,smoke,train,export,infer}.slurm`, `configs/slurm.example.env` | Deployment and Slurm lifecycle; T12 |
| `requirements/{candidate,uit-lock}.txt`, `README.md`, `tests/**` | Candidate/exact dependencies, command reference and behavioral verification; T01–T13 |

Shared records live in `contracts.py` with `schema_version=1`: `UtteranceRow`, `AuditResult`, `SnapshotIdentity`, `NoiseManifest`, `AugmentationResult`, `PreparedExample`, `RuntimeProfile`, `TrainingBudget`, `TrainingProgress`, `ArtifactIdentity`, `PreflightReport`, `SmokeReport`, `ExportResult`, and `OutputValidation`. They contain serializable metadata; waveform/tensor fields are excluded from persisted JSON records. `TrainingBundle` lives in `model.py` and holds the model, exact `target_names`, and runtime profile. Define each record's required fields when its owning task first consumes it; later tasks import the same type rather than redefining it.

## Task Execution Protocol

Read the task's specification sections and dependencies first. For T01–T12, use the listed failing behavioral tests, run them to confirm the intended failure, implement the smallest coherent deliverable, then run the same tests to pass. Record command/output evidence in [tasks.md](tasks.md). If execution occurs in a Git repository, commit each verified task using the task's suggested message; do not initialize a repository solely for this plan.

CPU commands below assume the future environment is installed and `asr` is the package console entry point. Model/GPU tests are marked separately and never run on the login node. All tasks begin unchecked. Existing checklist steps inherit their task's file list, dependencies, verification command and acceptance contract unless they specify a narrower scope; evidence always includes the command/result and artifact/report identity. Each refined subtask below specifies its objective, affected files, dependencies, verification, acceptance and evidence. Apply the same failing-test → implementation → passing-test cycle to each software subtask; record evidence before checking it in the tracker.

### T01: Locked configuration and package contract

**Depends on:** none. **Spec:** 1–3, 5, 15.
**Files:** create `.gitignore`, `pyproject.toml`, `configs/official.yaml`, `requirements/candidate.txt`, `src/asr/{__init__,__main__,config,contracts}.py`, `tests/{test_config,test_gitignore}.py`. T01.7 plans an operational assessment record under generated `manifests/`; it does not add a new project phase.
**Interfaces:** `load_config(path: Path) -> OfficialConfig`; `validate_locked(config: OfficialConfig) -> None`. `OfficialConfig` owns model/data/lora/training/augmentation/processor/inference/export settings and operational paths; frozen methodology values cannot be CLI-overridden.

- [ ] Write `test_locked_config_rejects_method_changes` and `test_operational_paths_do_not_change_method_hash`; parameterize locked fields, unknown keys, boolean-as-integer mistakes, and invalid operational paths.
- [ ] Run `pytest tests/test_config.py -q` and confirm failure before implementation.
- [ ] Implement validation, stable canonical JSON/method hash, and explicit operational path defaults from the specification. Copy the official YAML exactly; declare Python 3.11 and candidate stack versions without claiming UIT compatibility. Register the `gpu`, `snapshot` and `linux` pytest markers; real-platform tests are excluded from the portable CPU gate.
- [ ] Run the tests and verify these invariants:
  ```python
  cfg = load_config(Path("configs/official.yaml"))
  assert cfg.training.epochs == 10
  assert cfg.lora.scope == "encoder_decoder"
  assert cfg.data.deduplicate is False
  ```
- [ ] Record evidence and, when applicable, commit `feat: define locked ASR configuration`.

- [ ] **T01.6 — Exclude generated artifacts from Git.** Files: `.gitignore` and `tests/test_gitignore.py`; depends on T01.3 and the project-path contract.
  **Verification:** `pytest tests/test_gitignore.py -q` in an isolated temporary Git fixture, with tiny sentinel files instead of actual weights/audio. Run `git check-ignore --no-index` and a fixture `git add --all` / `git ls-files` check.
  **Acceptance:** ignore pretrained/model weight formats, `assets/pretrained/`, runtime noise payloads, `runs/`, checkpoints/temporary checkpoint directories, merged exports, outputs/transcripts/manifests, source dataset roots including `train+public_test/`, copied WAVs, venvs/caches and local credential-bearing configuration. Keep source, official configuration, example templates, dependency locks, tests and docs trackable. A tiny synthetic test WAV may be trackable only by an exact reviewed fixture allowlist; copied competition audio is never an allowed fixture.
  **Evidence:** list representative ignored and trackable paths with their Git results, and record a read-only check that forbidden paths are not already tracked in the actual repository. Ignore rules do not untrack existing files; report any violation without rewriting history or deleting data.

- [ ] **T01.7 — Record early UIT feasibility without blocking local implementation.** Files: generated `manifests/uit-feasibility.json`; T12 later consumes verified operational fields for `configs/slurm.example.env`. Depends on reading the specification, not on T12/T13 completion.
  **Verification:** review an operator-provided report or, only when access is available and authorized during execution, perform a lightweight read-only/CPU inspection of UIT paths, Slurm account/partition/QoS/GRES/MIG syntax, snapshot/noise availability, candidate dependency constraints and likely memory constraints.
  **Acceptance:** every field has a source/status of verified, unavailable or unknown, with a T13 follow-up; Windows access to UIT is never assumed. With no access, a truthful access-unavailable record satisfies the preparatory assessment attempt and local software work continues. Review the record before finalizing GPU integration in T05/T09; no GPU or full-model smoke is required by this early step.
  **Evidence:** dated assessment and concrete risks/blockers, with no credentials. Early findings cannot satisfy G2/G3 or substitute for T13. This planning revision itself performs no server inspection.


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
**Files:** create `src/asr/{assets,processor}.py`; `tests/{test_assets,test_processor}.py`; `tests/gpu/test_snapshot_mask_forward.py`.
**Interfaces:** `verify_snapshot(config: OfficialConfig) -> SnapshotIdentity`; `verify_noise(config: OfficialConfig) -> NoiseManifest`; `load_processor(snapshot: SnapshotIdentity) -> WhisperProcessor`; `prepare_example(row: UtteranceRow, audio: np.ndarray, processor: WhisperProcessor) -> PreparedExample`.
`NoiseManifest.entries` remains an ordered path/hash/license/review/non-speech list. `PreparedExample` contains `input_features`, integer frame `attention_mask`, `labels`, label padding information and utterance ID; frame and token masks are distinct.

**Processor contract:** validate the supplied audio view's length `n` before feature extraction: `0 < n <= 480000` at 16 kHz. Call the pinned processor with `padding="max_length"`, `max_length=480000`, `truncation=False`, `return_attention_mask=True` and `return_tensors="pt"`. Expect floating `input_features.shape == (1, 80, 3000)` and integer/binary `attention_mask.shape == (1, 3000)`; cast the latter to Torch long for model input. Frame `i` is valid when `160*i < n`, so its mask is 1, followed by 0 for padding. Use the actual augmented-view length, not the original WAV length. The mask describes sampled frame validity, not exact receptive-field support.

Pass this frame mask as `attention_mask` to the pinned Whisper forward/generation APIs; never substitute it for label padding or `decoder_attention_mask`, or manually downsample it to 1,500 encoder positions. The pinned forward path uses it in feature masking and does not pass it as an encoder self-attention padding mask. Test compatibility instead of promising padding-invariant encoder outputs. References: [feature extraction 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/whisper/feature_extraction_whisper.py), [Whisper forward 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/whisper/modeling_whisper.py).

- [ ] **T03.1 — Verify the exact local pretrained snapshot.** Files: `assets.py`, snapshot records in `contracts.py`, `tests/test_assets.py`; depends on T01 and T02 path containment.
  **Verification:** `pytest tests/test_assets.py -k snapshot -m "not gpu and not snapshot" -q` for CPU fixture tests (test names may include snapshot; the marker denotes real assets).
  **Acceptance/evidence:** missing files, incorrect repository/revision/hash, unsafe paths and incompatible architecture fail; complete fixture inventories pass. Save test results and a fixture snapshot identity. Actual snapshot inventory/tokenizer evidence is deferred to T13.

- [ ] **T03.2 — Verify the ordered noise asset manifest.** Files: `assets.py`, noise records in `contracts.py`, `tests/test_assets.py`; depends on T01/T02.
  **Verification:** `pytest tests/test_assets.py -k noise -m "not gpu and not snapshot" -q`.
  **Acceptance/evidence:** require nonempty licensed/reviewed non-speech entries with matching hashes, safe paths, correct PCM format and positive power; preserve order. Record failing-case IDs, passing test output and manifest hashes. Missing noise blocks admission, not implementation. `số_noise` / `số_frame_noise` retain their specification meanings.

- [ ] **T03.3 — Prepare padded features and explicit frame masks.** Files: `processor.py`, `tests/test_processor.py`, `tests/gpu/test_snapshot_mask_forward.py`; depends on T02 and T03.1.
  **Verification:** CPU fixtures test lengths 16000, 16001 and 480000, expecting valid mask sums 100, 101 and 3000, mask shape/dtype/values, unchanged source arrays and rejection of 480001 samples. Capture the padded waveform to prove the final valid sample is preserved. Run `pytest tests/test_processor.py -k "feature or mask" -m "not gpu and not snapshot" -q`.
  **Acceptance/evidence:** features/masks follow the contract exactly; no truncation or sample-rate change occurs. Implement `test_real_snapshot_mask_forward` accepting features plus the mask and producing finite loss/logits on the actual snapshot. Its actual command, `pytest tests/gpu/test_snapshot_mask_forward.py -m "gpu and snapshot" -q`, runs in T13's allocation; record CPU evidence now and leave real-model acceptance pending there.

- [ ] **T03.4 — Preserve the existing label/token contract.** Files: `processor.py`, `tests/test_processor.py`; depends on T03.1.
  **Verification:** `pytest tests/test_processor.py -k label -m "not gpu and not snapshot" -q` covers NFC + trim, decoder-start removal, 447 accepted / 448 rejected, empty text, retained EOS and shared PAD/EOS IDs.
  **Acceptance/evidence:** no training-target rewriting/truncation; only padding is masked with `-100`. Record boundary assertions and original-reference preservation; tokenize all real Train references only when the verified snapshot is available in T13.

- [ ] **T03.5 — Integrate and record asset/processor readiness.** Files: `assets.py`, `processor.py` and their records/tests; depends on T03.1–T03.4.
  **Verification:** `pytest tests/test_assets.py tests/test_processor.py -m "not gpu and not snapshot" -q`.
  **Acceptance/evidence:** one ordered asset identity feeds preparation without automatic downloads; all CPU tests pass and real-snapshot work is explicitly pending T13. Record evidence and commit when applicable, `feat: verify local assets and prepare Whisper inputs`.

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
**Interfaces:** `resolve_lora_targets(model: torch.nn.Module) -> list[str]`; `require_gpu_allocation(env: Mapping[str, str], hostname: str) -> None`; `probe_runtime(config: OfficialConfig, snapshot: SnapshotIdentity) -> RuntimeProfile`; `build_training_model(config: OfficialConfig, snapshot: SnapshotIdentity, runtime: RuntimeProfile) -> TrainingBundle`. `require_gpu_allocation` lives in `precision.py` and is reused by later GPU commands.

- [ ] **T05.1 — Resolve only the exact target module set.** Files: `model.py`, `tests/test_lora_targets.py`; depends on T01/T03.1.
  **Verification:** `pytest tests/test_lora_targets.py -k target -m "not gpu" -q` with complete and deliberately incorrect topology fixtures.
  **Acceptance/evidence:** exactly the six specified module families over 32 layers resolve to 192 names; missing/extra/out-of-scope projections fail. Record the target-set assertions; the actual snapshot count remains a T13 check.

- [ ] **T05.2 — Initialize the generic Whisper-compatible adapter.** Files: `model.py`, `tests/test_lora_targets.py`; depends on T05.1.
  **Verification:** `pytest tests/test_lora_targets.py -k "adapter or wrapper" -m "not gpu" -q` verifies seed-42 initialization, `input_features` forwarding and trainable-name/parameter assertions.
  **Acceptance/evidence:** generic `PeftModel`, fresh rank-8/alpha-16/dropout-0.05 adapter, frozen base/bias, no extra saved modules and no quantization. Expected actual A/B count is 3,932,160; save fixture counts and scope evidence without claiming real-model certification.

- [ ] **T05.3 — Guard GPU entry and resolve compute/storage precision.** Files: `precision.py`, `tests/test_precision.py`; depends on T01/T03.1. Review T01.7's feasibility record; unavailable server access does not block CPU implementation.
  **Verification:** `pytest tests/test_precision.py -m "not gpu" -q` covers missing allocation/login-node rejection, BF16 probe success/failure and FP16 + GradScaler fallback using controlled hardware fixtures.
  **Acceptance/evidence:** no CUDA initialization before guards, frozen base/A/B/AdamW storage strategy remains FP32, compute uses the selected autocast dtype, and profiles record probe/backend facts. Save CPU results; real kernel/memory probes remain T13 work.

- [ ] **T05.4 — Preserve gradient flow with checkpointing.** Files: `model.py`, `precision.py` and their tests; depends on T05.2–T05.3.
  **Verification:** a tiny CPU Whisper/LoRA fixture runs forward/backward with non-reentrant checkpointing and `use_cache=False`; compare frozen-base bytes and assert finite adapter gradients. Run `pytest tests/test_lora_targets.py -k gradient -m "not gpu" -q`.
  **Acceptance/evidence:** neither encoder nor decoder trainable execution is wrapped in `no_grad`; initial zero A gradients are permitted, and A connectivity is demonstrated after B changes at a nonzero LR. Save CPU gradient evidence; actual-model verification remains T13.

- [ ] **T05.5 — Integrate model initialization and runtime metadata.** Files: `model.py`, `precision.py` and their tests; depends on T05.1–T05.4.
  **Verification:** `pytest tests/test_lora_targets.py tests/test_precision.py -m "not gpu" -q`.
  **Acceptance/evidence:** `TrainingBundle` exposes the model, exact `target_names` and runtime profile, with no optimizer created here. Record passing CPU evidence and explicit pending GPU assertions; commit when applicable, `feat: initialize exact encoder decoder LoRA scope`.

### T06: Explicit training windows and scheduler

**Depends on:** T01–T05. **Spec:** 9, 11, 16.
**Files:** create `src/asr/{training,schedule}.py`; `tests/{test_accumulation,test_schedule,test_training_guards}.py`.
**Interfaces:** `compute_budget(n: int) -> TrainingBudget`; `epoch_order(n: int, epoch: int, seed: int = 42) -> list[int]`; `create_optimizer(bundle: TrainingBundle, config: OfficialConfig) -> torch.optim.AdamW`; `create_scheduler(optimizer: torch.optim.Optimizer, budget: TrainingBudget) -> torch.optim.lr_scheduler.LambdaLR`; `run_window(bundle, optimizer, scheduler, scaler, examples: Sequence[PreparedExample], progress: TrainingProgress) -> WindowResult`. `WindowResult` records loss/LR/norm/scaler state, whether the optimizer was invoked and whether the step succeeded.

`TrainingProgress` records `epoch`, `completed_epochs`, `next_position`, `global_micro_step`, `global_update`, `permutation_hash`, `pending_micro_steps`, `successful_updates_by_epoch` and `utterances_completed_by_epoch`. The two per-epoch arrays have 10 entries, start at 0 and advance only after successful windows; terminal values are 315 and 5,036 per entry for this dataset. At persisted boundaries `pending_micro_steps=0` and gradients are cleared with `zero_grad(set_to_none=True)`. Save the terminal marker only after the tenth epoch completes; canonical terminal state is `epoch=10`, `completed_epochs=10` and `next_position=0`. T10 validates semantic completion and also recognizes a fully completed last-epoch-index representation. `TrainingBudget` keeps the existing utterance/update/warmup fields.

**Scheduler convention:** `u` is the number of successful optimizer calls already completed, not a one-based epoch or micro-step. Initialize AdamW's base LR to `1e-4` and construct `LambdaLR(last_epoch=-1)` so its initial step installs `lambda(0)=0`. The LR used by the next optimizer call is `base_lr * lambda(u)`, where `lambda(u)=u/W` for `0 <= u < W` and `max(0, (U-u)/(U-W))` thereafter. After a successful call, increment progress and advance the scheduler exactly once. For the audited run, `U=3150` and `W=158`.

| Call / boundary | LR actually used or installed |
|---|---|
| First call, `u=0` | 0; parameters remain unchanged, but finite non-skipped AdamW execution advances its moment/step state and counts as one successful optimizer call |
| Second call, `u=1` | `1e-4 / 158` |
| Last call below the peak, `u=157` | `1e-4 * 157 / 158` |
| Warmup peak call, `u=158` (159th call) | `1e-4` |
| First decayed call, `u=159` | `1e-4 * 2991 / 2992` |
| Final call, `u=3149` | `1e-4 / 2992`, strictly positive |
| State after final call | `global_update=3150`, `scheduler.last_epoch=3150`, next LR 0; no additional optimizer call |

A “successful optimizer update” in this plan means an actual, finite, non-skipped optimizer call; it does not require a nonzero parameter delta. Zero-LR calls are not GradScaler skips. Detect skips through verified optimizer invocation/scaler behavior, never through the optimizer's return value, which can be `None` for successful calls. On failure do not advance the scheduler or completed progress. Checkpoint at a clean boundary; resume restores the installed next LR and scheduler/optimizer state without an extra initialization/step.

- [ ] **T06.1 — Resolve budget and deterministic epoch order.** Files: `training.py`, progress/budget records, `tests/test_accumulation.py`; depends on T01/T02.
  **Verification:** tests assert budget `(315, 3150, 158)` for 5,036 rows, 314 windows of 16 + one of 12, and reproducible seed-`42+epoch` permutations without omitted/repeated occurrences.
  **Acceptance/evidence:** `compute_budget` and `epoch_order` preserve locked totals/order semantics; save assertions and test output.

- [ ] **T06.2 — Implement and restore the exact LR timeline.** Files: `schedule.py`, `tests/test_schedule.py`; depends on T06.1.
  **Verification:** `pytest tests/test_schedule.py -q` tests `u=0,1,157,158,159,3149` and post-final state using the formula above. Round-trip optimizer/scheduler state immediately before/after the warmup peak and before the last call; compare the entire remaining LR trace with uninterrupted execution.
  **Acceptance/evidence:** construction installs 0, no off-by-one scheduler call occurs and resume changes no LR values. Save the boundary table and deterministic uninterrupted/resumed traces.

- [ ] **T06.3 — Construct A/B-only AdamW and define step success.** Files: `training.py`, `tests/test_training_guards.py`; depends on T05 and T06.2.
  **Verification:** a CPU AdamW test supplies finite nonzero gradients at LR 0, then checks unchanged parameters, advanced optimizer moments/step counters and `global_update==1`. Separate mocked GradScaler success/skip cases both return `None` but only the invoked step counts.
  **Acceptance/evidence:** only A/B enter the optimizer; LR/betas/epsilon/weight decay remain locked. Non-finite or skipped steps fail without budget/scheduler advancement; save success/skip evidence.

- [ ] **T06.4 — Accumulate and clip complete/partial windows.** Files: `training.py`, `tests/test_accumulation.py`; depends on T03–T05 and T06.1–T06.3.
  **Verification:** `pytest tests/test_accumulation.py -q` compares windows of 16 and 12 against the mean of individual utterance losses, tests fixed scaler scale throughout each window, and checks unscale → clip → optimizer → scheduler ordering.
  **Acceptance/evidence:** each loss is divided by actual `k`, clipping uses norm 1.0, and there is no carryover/drop-last or token-weighted substitute objective. Save gradient/update comparisons.

- [ ] **T06.5 — Persist meaningful progress and enforce failure guards.** Files: `training.py`, progress records and `tests/test_training_guards.py`; depends on T06.1–T06.4.
  **Verification:** `pytest tests/test_accumulation.py tests/test_schedule.py tests/test_training_guards.py -q` covers OOM, non-finite loss/gradients, failed/scaler-skipped calls, epoch transitions and clean terminal progress.
  **Acceptance/evidence:** counters advance only on successful windows, zero LR is logged explicitly, and no failed attempt is mistaken for completion. Record passing CPU evidence and commit when applicable, `feat: train with exact accumulation and fixed update budget`.

### T07: Immutable checkpoints and exact progress restore

**Depends on:** T01, T04–T06. **Spec:** 10–11, 16.
**Files:** create `src/asr/checkpoint.py`; `tests/{test_checkpoint,test_resume}.py`.
**Interfaces:** `save_checkpoint(run_dir: Path, attempt_id: str, state: CheckpointState) -> Path`; `load_checkpoint(path: Path, expected: ArtifactIdentity) -> CheckpointState`; `restore_training(state: CheckpointState, objects: TrainingObjects) -> TrainingProgress`. `CheckpointState` and `TrainingObjects` retain adapter, optimizer/scheduler/scaler, RNG, progress, identities and permutation; metadata records checkpoint kind, file hashes, successful-update counts and the exact final adapter identity when terminal.

- [ ] **T07.1 — Publish only complete immutable checkpoints.** Files: `checkpoint.py`, `tests/test_checkpoint.py`; depends on T01/T06.
  **Verification:** tests interrupt each publication stage, collide destinations and attempt concurrent writers; verify a same-filesystem temporary directory is not discoverable as resumable until all hashes/metadata exist and atomic rename succeeds.
  **Acceptance/evidence:** no overwrite or partial publication; unique attempt UUIDs are independent of Slurm job ID. Save publication/collision test output.

- [ ] **T07.2 — Serialize every required state at a clean boundary.** Files: `checkpoint.py`, `tests/test_checkpoint.py`; depends on T04–T06 and T07.1.
  **Verification:** round-trip adapter, optimizer, scheduler, scaler, Python/NumPy/Torch/CUDA RNG metadata, permutation and progress; reject `pending_micro_steps != 0` or pending gradients. Include terminal metadata and final adapter digest fixtures.
  **Acceptance/evidence:** restored fields and hashes match exactly where serialization is lossless; no hidden scheduler advance or unfinished window is saved. Record field-level comparisons.

- [ ] **T07.3 — Validate identity and restore objects before RNG.** Files: `checkpoint.py`, `tests/test_resume.py`; depends on T07.2 and T05.
  **Verification:** corrupt each protected identity, reconstruct objects that consume RNG and restore RNG last. Reject mismatch before any resumed optimizer call; assert restored next LR and cursor are unchanged.
  **Acceptance/evidence:** methodology/input/runtime mismatch fails, while equivalent operational path/attempt changes remain permitted under the existing identity policy. Save mismatch diagnostics and RNG/next-LR assertions.

- [ ] **T07.4 — Prove uninterrupted/resumed equivalence.** Files: `tests/test_resume.py` and checkpoint integration; depends on T07.1–T07.3.
  **Verification:** `pytest tests/test_checkpoint.py tests/test_resume.py -q` covers mid-epoch, partial-window boundary, epoch transition, warmup boundary and final-call resume. Compare IDs, augmentation digests, LR trace, adapter/optimizer state and final budget.
  **Acceptance/evidence:** no lost/double-counted committed progress and no incomplete checkpoint selected; save deterministic traces and state comparisons. Record passing CPU evidence and commit when applicable, `feat: preserve complete training state across resume`.

### T08: Run lifecycle, logs, admission reports, and CLI

**Depends on:** T01–T07. **Spec:** 6, 11, 14.1, 14.4, 16.
**Files:** create `src/asr/{artifacts,logging,preflight,cli}.py`; `tests/{test_artifacts,test_preflight,test_cli,test_run_loop,test_logging}.py`; extend `src/asr/__main__.py` and `src/asr/training.py`.
**Interfaces:** retain `create_run(config: OfficialConfig, identity: ArtifactIdentity, run_id: str) -> Path`; `start_attempt(run_dir: Path) -> str`; `run_preflight(config: OfficialConfig, stage: Literal["cpu", "gpu"], output: Path) -> PreflightReport`; `check_admission(config: OfficialConfig, preflight: Sequence[PreflightReport], smoke: SmokeReport) -> None`; `execute_run(config: OfficialConfig, run_dir: Path, resume_from: Path | None = None) -> TrainingProgress` in `training.py`; `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **T08.1 — Own run identity, attempts and lifecycle transitions.** Files: `artifacts.py`, `tests/test_artifacts.py`; depends on T01/T07.
  **Verification:** `pytest tests/test_artifacts.py -q` rejects duplicate run IDs, conflicting ownership and forbidden lifecycle transitions, while preserving prior attempts/checkpoints.
  **Acceptance/evidence:** method/input/runtime identities remain distinct from operational names/paths; no resumed attempt overwrites an earlier artifact. Save lifecycle/ownership test results.

- [ ] **T08.2 — Emit complete structured training and augmentation logs.** Files: `logging.py`, `tests/test_logging.py`; depends on T04/T06/T08.1.
  **Verification:** `pytest tests/test_logging.py -q` asserts timezone timestamps, run/attempt IDs, progress, raw/scaled loss, actual LR, window size, gradient norm, scaler/dtype/memory fields and every augmentation fallback.
  **Acceptance/evidence:** JSONL records are parseable and attributable to a successful step or a recorded failure; aggregate clean/aug/fallback totals reconcile. Save fixture logs and assertions.

- [ ] **T08.3 — Generate admission reports from existing validators.** Files: `preflight.py`, `tests/test_preflight.py`; depends on T02/T03/T05/T08.1.
  **Verification:** `pytest tests/test_preflight.py -q` covers stale/tampered/missing CPU/GPU/smoke reports, protected-hash changes, tokenizer failures and allocation guards before CUDA.
  **Acceptance/evidence:** `train` admission requires matching reports; CPU fixtures cannot manufacture actual GPU success. Missing external assets remain explicit operational blockers. Save report schema/identity tests.

- [ ] **T08.4 — Connect epoch/window orchestration to checkpointing.** Files: `training.py`, `tests/test_run_loop.py`; depends on T04–T07 and T08.1–T08.3.
  **Verification:** `pytest tests/test_run_loop.py -q` traverses 10 tiny-model epochs, resumes mid-epoch, requests a boundary stop and tests update-50/epoch/final checkpoint events.
  **Acceptance/evidence:** `execute_run` honors permutation/cursor, invokes one view per occurrence, clears completed windows and persists coherent progress. Shared-boundary events record all reasons; a separate terminal checkpoint identity is established after full epoch-10 completion and the final adapter is linked to it. Save visitation/checkpoint-event traces.

- [ ] **T08.5 — Expose guarded CLI entry points and integration.** Files: `cli.py`, `__main__.py`, `tests/test_cli.py`; depends on T08.1–T08.4.
  **Verification:** `pytest tests/test_artifacts.py tests/test_logging.py tests/test_preflight.py tests/test_cli.py tests/test_run_loop.py -q` checks audit/preflight/train/resume routing and nonzero diagnostic exits.
  **Acceptance/evidence:** fresh official initialization and resume stay distinct, and no train route bypasses admission. Later commands remain unavailable until their owning tasks wire them. Record passing CPU evidence and commit when applicable, `feat: gate ASR runs with identity bound preflight reports`.

### T09: Full-model smoke and continuation verification

**Depends on:** T01–T08. **Spec:** 14.2–14.3.
**Files:** create `src/asr/smoke.py`, `tests/test_smoke.py`, `tests/gpu/test_full_model_smoke.py`; extend `src/asr/cli.py`.
**Interfaces:** `run_smoke(config: OfficialConfig, audit: AuditResult, reports: Sequence[PreflightReport], run_id: str) -> SmokeReport`. `SmokeReport` exposes fixture IDs, protected identities, profile, checks/tolerances, artifact hashes and `passed`.

- [ ] Write CPU harness tests rejecting warm-start use and fake GPU success; write a separately marked real-model smoke test comparing uninterrupted and resumed execution from the same initial adapter.
- [ ] Run `pytest tests/test_smoke.py -q`; confirm failure. GPU tests stay pending until T13; skipped tests are not passes.
- [ ] Review the nonblocking T01.7 feasibility record before finalizing GPU integration, then implement an isolated `smoke-` run with 44 manifest occurrences in windows 16, 16, 12, using the official LR schedule/budget with a smoke-only execution limit. Save/resume after the first window; this does not change the official config or its epoch budget. Include a maximum-shape 30-second/447-label technical probe, allocation checks and memory measurements.
- [ ] Run CPU tests and implement the separately marked GPU acceptance check: sample/augmentation/LR/progress equality, base-weight equality, correct dtype/scaler behavior and adapter-state comparison at `rtol=1e-4` / `atol=1e-5` in the same environment. Verify A gradient connectivity after B has changed at a nonzero LR. Actual execution of `pytest tests/gpu/test_full_model_smoke.py -m gpu -q` occurs in T13 inside an allocation; failed comparisons block admission and must not trigger automatic tolerance adjustment.
- [ ] Complete this software task with passing CPU evidence and the real-model test/harness ready to run; leave actual GPU execution/report acceptance pending T13/G3. Never synthesize a passing `SmokeReport`. Commit code when applicable, `feat: verify full model training checkpoint and resume`.

### T10: Verified final adapter merge and FP16 export

**Depends on:** T03, T05, T07–T08. **Spec:** 12, 17.
**Files:** create `src/asr/export.py`, `tests/test_export.py`, `tests/gpu/test_export_reload.py`; extend `src/asr/cli.py`.
**Interfaces:** retain `export_final(config: OfficialConfig, checkpoint: Path, output_dir: Path) -> ExportResult`. Add `verify_terminal_checkpoint(state: CheckpointState, budget: TrainingBudget, expected: ArtifactIdentity) -> None` in `export.py`. `ExportResult` retains identities, file hashes, dtype inventory, numerical checks, reload result and `valid`.

**Terminal-checkpoint contract:** internal index `9` identifies the tenth epoch, not completion. Reject an ordinary checkpoint taken during that epoch when `completed_epochs < 10`, even if its directory is named final. Accept export only after verifying `completed_epochs == 10`, `global_update == budget.total_updates`, all expected successful epoch/window counts, `global_micro_step == 10 * budget.utterances_per_epoch`, `pending_micro_steps == 0`, no unsaved gradients, terminal checkpoint metadata, and the final adapter's actual hash/scope/base/run identity.

Canonical terminal progress is `epoch=10` and `next_position=0`. A semantically equivalent terminal record retaining `epoch=9` and `next_position=budget.utterances_per_epoch` is also legitimate when every terminal invariant is proven. Export must not reject it solely for storing the last zero-based index. The checkpoint and adapter identities, not filenames alone, establish finality. These representation rules do not allow early export or selection of an intermediate checkpoint.

- [ ] **T10.1 — Verify semantic terminal completion.** Files: `export.py`, `tests/test_export.py`; depends on T06/T07.
  **Verification:** `pytest tests/test_export.py -k terminal -m "not gpu" -q` accepts canonical terminal state and the fully completed index-9 representation. Reject index 9 with only 9 completed epochs, 3,149 updates, a pending partial window/gradient, incomplete visitation counts, a mismatched adapter digest and a falsely named final directory.
  **Acceptance/evidence:** both legitimate terminal representations pass; no incomplete state passes. Record the acceptance/rejection matrix with the expected 3,150 updates and 50,360 micro-steps for this dataset.

- [ ] **T10.2 — Bind merge inputs and protect destinations.** Files: `export.py`, `tests/test_export.py`; depends on T03/T05/T10.1.
  **Verification:** tests reject base/revision/run/adapter identity mismatches and existing or incomplete export destinations before merge/publication.
  **Acceptance/evidence:** only the verified final adapter and exact pretrained base are consumed; no destination is overwritten. Save identity/destination diagnostics.

- [ ] **T10.3 — Merge and publish the complete FP16 artifact.** Files: `export.py`, export records, `tests/test_export.py`; depends on T10.2.
  **Verification:** tiny-model tests merge FP32 base/adapter in eval mode, convert to FP16 and check finite weights, wrapper removal, complete processor/tokenizer/generation metadata and hashes.
  **Acceptance/evidence:** an atomic complete export contains all required local files, provenance and the terminal adapter hash. Save the tiny-model export inventory and numerical report.

- [ ] **T10.4 — Verify fresh-process offline reload.** Files: `export.py`, `tests/test_export.py`, `tests/gpu/test_export_reload.py`; depends on T10.3.
  **Verification:** `pytest tests/test_export.py -m "not gpu" -q` plus a separately marked real-model check compares reference/merged logits at the declared `atol=0.05` / `rtol=0.05`, checks FP16 floating weights and local-only generation without PEFT.
  **Acceptance/evidence:** CPU tiny-model mechanics pass now; actual final-model reload/comparison is pending T14/G5. No automatic tolerance adjustment or tiny-model certification of real weights.

- [ ] **T10.5 — Wire final-only export and record evidence.** Files: `cli.py`, `export.py`, `tests/test_export.py`; depends on T10.1–T10.4.
  **Verification:** CLI tests show `merge-export` always invokes terminal/identity validation before model loading or output publication.
  **Acceptance/evidence:** all CPU export tests pass, real-model checks are explicitly pending, and intermediate checkpoints cannot bypass the guard. Commit when applicable, `feat: export the final merged FP16 Whisper model`.

### T11: Offline Train transcription and completeness validation

**Depends on:** T02–T03, T08, T10. **Spec:** 13–14, 17.
**Files:** create `src/asr/{inference,outputs}.py`; `tests/{test_inference,test_inference_resume,test_outputs}.py`; extend `src/asr/cli.py`.
**Interfaces:** preserve `transcribe_train(config: OfficialConfig, manifest: Path, exported: ExportResult, output_dir: Path) -> OutputValidation`; `group_transcripts(rows: Sequence[UtteranceRow], predictions: Sequence[dict]) -> list[dict]`; `validate_outputs(manifest: Path, utterance_file: Path, source_file: Path, exported: ExportResult) -> OutputValidation`. Continuation is internal to `transcribe_train`; no new public output schema or top-level command is introduced.

**Durable continuation contract:** keep `output_dir/.inference-progress/identity.json` and `results.partial.jsonl` under the existing run output directory. Atomically establish identity before the first result, binding the verified merged-model file inventory, run/model identity, effective inference configuration, exact ordered utterance manifest/audio identities and fully resolved generation configuration/version. Exclude attempt IDs/timestamps from identity equality. One process-scoped writer lock protects this directory and is released on process exit; verify actual Linux filesystem locking in T13.1.

Each committed journal line is one successful, nonempty, completed utterance prediction in the existing utterance-record schema, in exact manifest order. Write the complete UTF-8 JSON line plus newline, flush and `fsync` before advancing; infer the next index from the durable prefix rather than a separate cursor file. If killed before commit, recomputing that utterance is allowed, but it must produce only one committed record. Empty/error/token-limit-without-EOS results retain token/error evidence in a diagnostic report and stop at the current index; never mark them completed or substitute references.

On restart, acquire the writer lock, compare every protected identity and validate each committed prefix row's ID/path/audio hash/model/status against the corresponding manifest row. Reject mismatches, duplicate/out-of-order records, corruption in any newline-terminated record or prefix lengths beyond the manifest. Recover only an unterminated trailing fragment by recording the recovery and truncating that private journal back to its last committed newline; never edit source data or silently repair a complete corrupt record. Repeating `infer-train` continues a compatible interrupted journal; an incompatible journal requires a new output/run location and never merges results.

After complete prefix coverage, build the public utterance/source files as private candidates, validate their unchanged schemas/mappings/counts and save candidate hashes. Publish each file atomically, then publish a valid `output-validation.json` completion report last. If interrupted between these operations, resume verifies existing published files against candidate hashes and finishes only the missing matching steps; it never overwrites an unknown/different completed artifact. Consumers require a valid completion report with matching file hashes. An existing fully valid result is an identity-checked no-op, not a new inference pass.

- [ ] **T11.1 — Resolve and execute locked offline generation.** Files: `inference.py`, `tests/test_inference.py`; depends on T03/T10.
  **Verification:** `pytest tests/test_inference.py -q` tests inherited sampling/timestamp/prefix settings, eval/inference mode, original audio and frame-mask forwarding, token evidence and the 440-token boundary.
  **Acceptance/evidence:** `do_sample=False`, `num_beams=1`, Vietnamese transcription, `max_new_tokens=440` and explicit no-temperature-fallback behavior stay locked; references/assistant/test data never become predictions. Save resolved-generation fixtures and assertions.

- [ ] **T11.2 — Persist and resume an identity-safe ordered prefix.** Files: `inference.py`, `tests/test_inference_resume.py`; depends on T08/T11.1.
  **Verification:** `pytest tests/test_inference_resume.py -q` interrupts before/after newline and `fsync`, handles an unterminated tail, rejects a complete corrupt line, duplicate/out-of-order IDs, concurrent writers and each model/config/manifest/generation mismatch, and resumes a fully committed prefix.
  **Acceptance/evidence:** resumed output order/coverage equals uninterrupted output without repeated committed rows; only committed successes are skipped. Save crash-injection/prefix comparisons and identity diagnostics. Actual Linux lock/flush semantics remain a T13.1 check, not an assumed Windows result.

- [ ] **T11.3 — Group source records using original history indices.** Files: `outputs.py`, `tests/test_outputs.py`; depends on T02/T11.1.
  **Verification:** `pytest tests/test_outputs.py -k grouping -q` covers no history, interleaved assistant turns and multiple user turns.
  **Acceptance/evidence:** exactly one source record per source, original JSONL/source order, correct current and `history_index` mapping, and unchanged public output schema. Save expected-versus-actual fixture records.

- [ ] **T11.4 — Validate complete coverage before final publication.** Files: `outputs.py`, `inference.py`, `tests/{test_outputs,test_inference_resume}.py`; depends on T11.2–T11.3.
  **Verification:** tests inject missing/extra/duplicate/wrong-identity rows, empty/incomplete predictions, crashes between the two public file installs and a missing/stale completion report.
  **Acceptance/evidence:** all 5,036/1,545 expected records are required for the actual dataset; failures remain explicit. Matching interrupted publication finishes idempotently; unknown existing artifacts fail. Save candidate/file hashes and output-validation assertions.

- [ ] **T11.5 — Integrate continuation into CLI and lifecycle.** Files: `cli.py`, `inference.py`, `outputs.py` and their tests; depends on T11.1–T11.4.
  **Verification:** `pytest tests/test_inference.py tests/test_inference_resume.py tests/test_outputs.py -q` exercises repeated `infer-train` and `validate-output`; lifecycle reaches `transcripts_complete` only after a valid final report.
  **Acceptance/evidence:** restart uses the same public command and locked settings, output schemas remain unchanged, and failed/incomplete inference is never advertised as complete. Record passing CPU evidence and commit when applicable, `feat: generate and validate mapped Train transcripts`.

### T12: UIT environment setup and Slurm lifecycle

**Depends on:** T01, T05, T07–T11 software deliverables, not T13 or a completed G1 Linux check. **Spec:** 3, 10, 14–16.
**Files:** create `scripts/{setup_env,submit_job}.sh`, `scripts/{preflight,smoke,train,export,infer}.slurm`, `configs/slurm.example.env`, `tests/test_slurm_scripts.py`; create/extend `README.md`. Linux-specific publication/locking tests stay in T07/T11's existing test files.
**Interfaces:** `submit_job.sh <preflight|smoke|train|resume|export|infer> [CLI arguments]` validates resource settings, then invokes `sbatch` with explicit command-line resource/output flags. Slurm scripts invoke the shared CLI; `resume` uses `train.slurm` with a checkpoint argument.

**Platform completion contract:** T12 software completion requires portable fake-scheduler tests and a documented runnable Linux verification procedure. G1 tracks `G1.cpu` and `G1.linux` evidence under the same main gate. If WSL/Linux is available, run the per-file Bash syntax loop and `pytest tests/test_slurm_scripts.py tests/test_inference_resume.py tests/test_checkpoint.py -m linux -q` there, recording script/source hashes and actual executed test counts. Require real POSIX warning-signal forwarding, writer-lock release after process termination, journal flush/torn-tail recovery and checkpoint publication tests; a Windows mock or skipped test does not satisfy this evidence.

If Linux is unavailable locally, mark G1.linux pending and proceed only to the explicitly CPU-only commissioning entry T13.1. That step runs the same checks on UIT before any GPU stage; it cannot claim G1 complete until they pass. T13.1 also verifies lock/flush/rename behavior on the actual project filesystem even if WSL already established script syntax. Access/credentials are operational prerequisites, not assumed Windows capabilities. Full G1 and G2 are required before T13.4/G3 and official operation.


- [ ] Write tests using fake `sbatch`/`srun` executables for missing account/partition/GRES, paths containing spaces, argument forwarding, log directories, exit status propagation, signals and a restarted job with the same Slurm job ID.
- [ ] Run `pytest tests/test_slurm_scripts.py -q`; confirm failure.
- [ ] Implement one task/process/GPU instance, project-local venv/caches/temp/logs, and explicit project paths. Read Slurm resources from operational settings and pass them to `sbatch`; do not interpolate shell variables inside `#SBATCH` directives. Use `srun` for the Python job step and verify the configured warning signal reaches the training process. Create log parents before submission.
- [ ] Run portable script tests; execute `for script in scripts/*.sh scripts/*.slurm; do bash -n "$script" || exit; done` and the Linux-marked tests on WSL/Linux when available, otherwise record them as pending G1.linux/T13.1. T12 software completion does not require an unavailable platform or a prior T13 pass. Document the warning signal, measured-lead-time requirement, allocation guards, CPU versus GPU commands, setup, explicit snapshot preparation, noise review, train, resume, export and inference. Scripts must never auto-submit official training after another job.
- [ ] Record portable CPU evidence, actual Linux results or an explicit G1.linux deferral, and commit when applicable, `feat: run the ASR lifecycle through configurable Slurm jobs`. README explains repeating `infer-train` with the same identities after interruption and requiring its valid completion report.

### T13: UIT commissioning and preflight evidence

**Depends on:** T01–T12 software readiness and G1.cpu; G1.linux may be pending only at CPU-only T13.1 entry. **Spec:** 3–5, 7–9, 14–16.
**Files:** create `requirements/uit-lock.txt` only after actual environment verification; update `README.md` and operational `configs/slurm.example.env` without changing the official method. Reports go under `manifests/` and isolated `runs/smoke-<id>/`.
**Interfaces:** use the CLI and scripts from T08–T12. Produce identity-bound CPU/GPU `PreflightReport` records and a passing `SmokeReport` for the actual server inputs/environment.

- [ ] **T13.1 — CPU-only commissioning entry and Linux verification.** Revisit the T01.7 assessment without assuming access; inspect authorized UIT paths/resource configuration and the recorded/advertised MIG profile read-only, without accessing `CS406/`. Run T12's Bash syntax/Linux-marked verification, recording actual project-filesystem lock/flush/rename behavior and report/script hashes. Close G1.linux only on actual passes; missing Linux/access remains a named blocker. This entry does not initialize CUDA, load GPU models or require G1.linux to have passed beforehand. Actual allocated GPU visibility/driver facts are established in T13.4.
- [ ] Prepare the isolated Python 3.11 environment and CPU-test dependencies as needed for T13.1, then the exact local pretrained snapshot and independently licensed/reviewed non-speech noise. No snapshot/noise/GPU result is needed merely to enter the CPU-only step. Runtime still has no automatic downloads; do not fabricate licensing/review evidence. Produce the exact dependency lock after compatibility checks.
- [ ] Run the CPU gate:
  ```bash
  asr audit --config configs/official.yaml --data-root "$DATA_ROOT" --output manifests/train
  asr preflight --config configs/official.yaml --data-root "$DATA_ROOT" --stage cpu --output manifests/preflight-cpu.json
  ```
  Operational paths resolve from the same configuration and `DATA_ROOT`; expected audit is 1,545/5,036, exact Train hash, no unresolved errors and all label lengths <=447.
- [ ] After full G1 and G2 pass, submit only GPU preflight and smoke through the wrapper, with official inputs and the isolated smoke run ID. Run T03's actual-snapshot feature/mask forward test inside the allocation; record finite loss/logits, accepted frame masks and allocated GPU/driver facts. Measure maximum-shape memory, window duration and checkpoint duration; set and test the early-warning lead time. Repeat relevant checks after any runtime/asset change.
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
- [ ] Run `infer-train` and `validate-output` on original audio through the GPU scripts; repeat the same inference command after interruption only with matching journal identities. Require all 5,036 utterances and 1,545 source records, correct IDs/indices/hashes, a valid completion report matching published hashes and no unresolved empty/truncated/error output.
- [ ] Finalize `run-manifest.json`, `run-summary.json` and `output-validation.json` with acceptance evidence. Mark only the stages actually achieved; preserve all prior checkpoints/attempts. An interrupted/blocked run remains incomplete.

## Dependency Order and Verification Gates

Keep the existing task order: T01 → T02 → T03 → T04/T05 → T06 → T07 → T08 → T09/T10 → T11 → T12 → T13 → T14. Slashes indicate independent dependencies, not delegation instructions. T01.7 is an early, nonblocking feasibility record before GPU integration is finalized, not a new phase or a substitute for commissioning.

T13 may start its CPU-only entry after T01–T12 software readiness and G1.cpu; pending G1.linux is resolved in T13.1 if no local Linux environment exists. Preparing the CPU environment needed to run those checks does not require snapshot/noise/GPU readiness. No GPU stage begins until full G1 and G2 pass. This separates software dependencies from platform evidence and removes the T12/T13 access cycle.

| Gate | Required evidence | Unlocks |
|---|---|---|
| G1: Implementation and Linux execution | G1.cpu: portable CPU tests/CLI/config/Git isolation pass. G1.linux: actual Bash syntax and POSIX signal/lock/publication tests pass, locally or in CPU-only T13.1 | G1.cpu permits CPU-only UIT commissioning; full G1 is required before GPU stages |
| G2: Input identity | Server Train hash/counts, all valid audio/labels, verified snapshot/noise manifests and dependency lock | GPU preflight only after full G1 also passes |
| G3: GPU readiness | Actual allocation/profile, targets/trainables, dtype/kernel/memory probes, real-snapshot frame-mask forward, uninterrupted/resumed smoke and actual Slurm signal delivery | Scheduling the official run |
| G4: Training completion | Fresh official initialization, exactly 10 completed epochs / 3,150 successful calls, no pending accumulation and verified terminal adapter/checkpoint identity | Final export |
| G5: Model export | Correct base/final adapter, merged FP16 artifact, fresh-process offline reload and numerical report | Official Train inference |
| G6: End-to-end output | Complete validated 5,036/1,545 mapping, identity-safe interrupted inference recovery, publication completion report and run summary with hashes | System acceptance |

G1.cpu uses `pytest tests -m "not gpu and not snapshot and not linux" -q`. Register all three markers; report excluded/skipped tests as pending. G1.linux uses T12's per-file Bash loop and actual Linux tests, not a mock or a source-only review. T01–T12 completion means their software and portable CPU deliverables are ready; actual snapshot/GPU tests belong to T13/G3, and final-model checks belong to T14/G5. Preserve all six main gates; the two G1 evidence components are not additional gates.

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

This revision updates planning only. Implementation begins in a later execution stage; no dependency installation, dataset modification, GPU access, Slurm submission or remote work is performed by this documentation change. T13 cannot be marked complete from local/stub tests, and T14 cannot be marked complete from a passing smoke test.

Review this plan and its conventions before implementation. Inline execution uses `executing-plans`; delegated execution uses `subagent-driven-development` only when selected. The tracker records evidence and blockers, and the specification remains authoritative if an implementation detail conflicts with a locked requirement.

