# PhoWhisper-large LoRA ASR Task Tracker

**Specification:** [specs.md](specs.md).  
**Implementation plan:** [plan.md](plan.md).  
**Status:** planning complete; all implementation and server-operation tasks below are not started.  
**Tracking rule:** this file owns completion status. Check an item only after its acceptance evidence exists; record commands, report paths/hashes and blockers in the execution record. A written test or a skipped GPU test is not a passing test.

## Scope and Execution Rules

The final locked configuration in specification section 5 applies to every task. Do not change the base/revision, encoder + decoder LoRA, seed, dataset, augmentation, 10 epochs, batch 1 / accumulation 16, optimizer/schedule, checkpoint selection or FP16 export.

Keep source data read-only and external to the Ubuntu project. Never access `CS406/`. Do not use Public/Private Test, environment state, tool labels or assistant turns for ASR training. Preserve every duplicate occurrence, fail on conflicts, and never silently skip invalid rows.

T01–T12 build and test software. T13 commissions the actual UIT environment. T14 operates the official run only after matching preflight/smoke evidence and operator scheduling. Creating these documents has not performed any of those tasks.

For each implementation task, follow the plan's failing-test → minimal implementation → passing-test steps. Record evidence, then commit the verified task when executing within an existing Git workflow. Do not initialize a repository or submit jobs merely to satisfy this tracker.

## Ordered Work

### T01 — Locked configuration and package contract

**Depends on:** none. **Deliverable:** package/configuration schema, unchanged official YAML and canonical method identity.  
**Verify:** `pytest tests/test_config.py -q`.

- [ ] T01.1 Write rejection tests for every locked-field override, invalid types/keys and unsafe operational paths.
- [ ] T01.2 Run the tests and record the intended failures before implementation.
- [ ] T01.3 Implement `OfficialConfig`, `load_config`, `validate_locked`, project packaging and candidate requirements.
- [ ] T01.4 Verify official epochs 10, scope `encoder_decoder`, deduplication false, and stable method hashes across permitted path changes.
- [ ] T01.5 Record passing evidence and checkpoint the change in the existing Git workflow when applicable.

### T02 — Safe projection and Train audit

**Depends on:** T01. **Deliverable:** read-only audit, stable utterance manifest, error and duplicate/conflict reports.  
**Verify:** `pytest tests/test_projection.py tests/test_audio_audit.py tests/test_paths.py -q`.

- [ ] T02.1 Write fixtures covering current/history index mapping, assistant exclusion, truncated/invalid WAVs, missing files, duplicate/conflicting PCM and symlink/path escapes.
- [ ] T02.2 Confirm the new tests fail for the intended missing behavior.
- [ ] T02.3 Implement safe resolution, PCM decoding/hashing, transcript normalization, source/utterance identity checks and audit outputs.
- [ ] T02.4 Verify duplicate retention, conflict failure, original order/index preservation, zero dataset writes and no environment/label exports.
- [ ] T02.5 Record passing evidence and commit when applicable.

### T03 — Asset verification and processor

**Depends on:** T01–T02. **Deliverable:** local snapshot/noise verification and non-truncating Whisper input preparation.  
**Verify:** `pytest tests/test_assets.py tests/test_processor.py -m "not gpu and not snapshot" -q`.

- [ ] T03.1 Write snapshot/noise hash, licensing/review metadata, containment and audio-format tests; add 447/448-token, decoder-start and PAD/EOS boundary tests.
- [ ] T03.2 Confirm failing tests before implementation.
- [ ] T03.3 Implement verified ordered asset manifests and `prepare_example` with 80 bins and explicit 3,000-frame padding.
- [ ] T03.4 Verify features `[1, 80, 3000]`, labels <=447, preserved EOS/content, padding `-100` and no automatic downloads.
- [ ] T03.5 Record CPU evidence; leave actual snapshot/tokenizer checks pending T13; commit when applicable.

### T04 — Deterministic E4 augmentation

**Depends on:** T02–T03. **Deliverable:** one deterministic waveform view per utterance/epoch, with complete decisions/fallback logs.  
**Verify:** `pytest tests/test_augmentation.py -q`.

- [ ] T04.1 Write independent seed/draw-trace fixtures and tests for all gates, speed/volume/noise transforms, noise wrapping, zero power and fallback boundaries.
- [ ] T04.2 Confirm failing tests before implementation.
- [ ] T04.3 Implement the exact SHA-256 seed, conditional draw order, interpolation convention, SNR formula and clean fallback behavior.
- [ ] T04.4 Verify seed `18138155238159038922` for `42:0:train_0001/current:augmentation-v1`, replay waveform equality, unchanged references and no redrawing/dataset doubling.
- [ ] T04.5 Record golden fixture provenance, NumPy/noise identities and passing evidence; commit when applicable.

### T05 — Exact LoRA and runtime precision

**Depends on:** T01, T03. **Deliverable:** generic Whisper-compatible PEFT model, exact trainable scope and measured runtime profile.  
**Verify:** `pytest tests/test_lora_targets.py tests/test_precision.py -m "not gpu" -q`; actual GPU assertions in T13.

- [ ] T05.1 Write exact-topology, scope-exclusion, generic `input_features`, allocation-guard and precision-selection tests.
- [ ] T05.2 Confirm failing CPU tests.
- [ ] T05.3 Implement fresh seed-42 base/adapter initialization, frozen FP32 base, FP32 A/B, autocast profile and non-reentrant checkpointing.
- [ ] T05.4 Verify model topology and guards; require actual 192 targets / 3,932,160 trainable A/B tensors on UIT, without assuming nonzero initial A gradients.
- [ ] T05.5 Record CPU evidence and pending GPU checks; commit when applicable.

### T06 — Training windows and scheduler

**Depends on:** T01–T05. **Deliverable:** exact optimizer budget, per-epoch ordering, partial windows and update/error handling.  
**Verify:** `pytest tests/test_accumulation.py tests/test_schedule.py tests/test_training_guards.py -q`.

- [ ] T06.1 Write gradient-equivalence, LR-boundary, visitation, scaling/clipping-order and failed-step guard tests for windows 16 and 12.
- [ ] T06.2 Confirm failing tests.
- [ ] T06.3 Implement budget/permutation functions, A/B-only AdamW, linear scheduler, loss/k accumulation and progress records.
- [ ] T06.4 Verify budget `315 / 3150 / 158`, final window 12, one successful optimizer/scheduler update per window and hard failure on OOM/non-finite/scaler skips.
- [ ] T06.5 Record passing evidence and commit when applicable.

### T07 — Checkpoints and resume

**Depends on:** T01, T04–T06. **Deliverable:** immutable, complete checkpoints and correct continuation after interruptions.  
**Verify:** `pytest tests/test_checkpoint.py tests/test_resume.py -q`.

- [ ] T07.1 Write tests for interrupted writes, collisions/concurrent writers, identity mismatches, partial-window/epoch transitions and RNG-consuming reconstruction.
- [ ] T07.2 Confirm failing tests.
- [ ] T07.3 Implement atomic publication, unique attempt ownership, adapter/optimizer/scheduler/scaler/RNG/progress/permutation state and RNG-last restoration.
- [ ] T07.4 Verify resumed IDs, augmentation, LR, adapter/optimizer state and budget match uninterrupted execution; incomplete checkpoints are never resumable.
- [ ] T07.5 Record passing evidence and commit when applicable.

### T08 — Lifecycle, admission, logs and CLI

**Depends on:** T01–T07. **Deliverable:** connected training/resume orchestration, identity-bound reports and guarded CLI.  
**Verify:** `pytest tests/test_artifacts.py tests/test_preflight.py tests/test_cli.py tests/test_run_loop.py -q`.

- [ ] T08.1 Write tests for stale/tampered reports, missing smoke, duplicate run IDs, forbidden states, complete logs, 10-epoch traversal and checkpoint triggers.
- [ ] T08.2 Confirm failing tests.
- [ ] T08.3 Implement run/attempt identities, `execute_run`, logging, CPU/GPU preflight and admission checks; wire audit/preflight/train/resume commands.
- [ ] T08.4 Verify official training is unreachable without matching CPU/GPU/smoke evidence, fresh initialization is separate from resume and checkpoints fire at the correct boundaries.
- [ ] T08.5 Record passing evidence and commit when applicable.

### T09 — Isolated full-model smoke

**Depends on:** T01–T08. **Deliverable:** smoke harness and actual-model continuation report, separate from the official run.  
**Verify:** `pytest tests/test_smoke.py -q`; later `pytest tests/gpu/test_full_model_smoke.py -m gpu -q` inside an allocation.

- [ ] T09.1 Write harness and real-model tests comparing uninterrupted/resumed execution, frozen base state and adapter gradient connectivity.
- [ ] T09.2 Confirm CPU harness tests fail; leave GPU execution pending until allocation exists.
- [ ] T09.3 Implement isolated windows 16, 16, 12, checkpoint/resume, the 30-second/447-label shape probe and runtime/memory recording.
- [ ] T09.4 Verify CPU tests and implement the GPU assertions for matching IDs/augmentation/LR/progress and adapter state within the declared tolerance; leave actual GPU execution to T13. No smoke weights may initialize official training.
- [ ] T09.5 Record passing CPU evidence and the ready-to-run GPU test/harness; commit code when applicable. A passing actual-server `SmokeReport` remains pending T13/G3.

### T10 — Final merge and FP16 export

**Depends on:** T03, T05, T07–T08. **Deliverable:** final-only export with offline reload and numerical verification.  
**Verify:** `pytest tests/test_export.py -m "not gpu" -q`; full-model checks after official training.

- [ ] T10.1 Write tests rejecting intermediate checkpoints, base/adapter mismatches, overwritten destinations, invalid weights and incomplete exports.
- [ ] T10.2 Confirm failing tests.
- [ ] T10.3 Implement FP32 base/adapter merge, FP16 export, processor/tokenizer/generation metadata, hashes and fresh-process offline reload.
- [ ] T10.4 Verify tiny-model mechanics and declared logit tolerances; keep actual final-model verification pending T14.
- [ ] T10.5 Record evidence and commit when applicable; never label a tiny-model check as final-model certification.

### T11 — Train inference and output validation

**Depends on:** T02–T03, T08, T10. **Deliverable:** original-audio inference and correct utterance/source transcript artifacts.  
**Verify:** `pytest tests/test_inference.py tests/test_outputs.py -q`.

- [ ] T11.1 Write mapping/coverage tests and generation tests for inherited sampling settings, empty output, EOS/token limits and incorrect identities.
- [ ] T11.2 Confirm failing tests.
- [ ] T11.3 Implement offline eval/inference-mode greedy generation, raw token evidence, source grouping and `validate_outputs`; wire infer-train/validate-output.
- [ ] T11.4 Verify original history indices/order, exact manifest coverage, recorded generation failures and absence of references/assistant/test data in outputs.
- [ ] T11.5 Record passing evidence and commit when applicable.

### T12 — Setup, Slurm scripts and README

**Depends on:** T01, T05, T07–T11. **Deliverable:** configurable deployment scripts, correct signals and complete operational documentation.  
**Verify:** `pytest tests/test_slurm_scripts.py -q` and `for script in scripts/*.sh scripts/*.slurm; do bash -n "$script" || exit; done` on Ubuntu.

- [ ] T12.1 Write fake-scheduler tests for required resources, argument quoting, logging, exit status, repeated job IDs and signal delivery.
- [ ] T12.2 Confirm failing tests.
- [ ] T12.3 Implement setup/submission wrappers and preflight/smoke/train/export/infer job scripts, with all generated paths under the project.
- [ ] T12.4 Verify shell syntax, resource forwarding, early-warning handling and allocation guards; document audit/setup/preflight/smoke/train/resume/export/inference/output commands.
- [ ] T12.5 Record passing evidence and commit when applicable; scripts must not chain into automatic official training submission.

### T13 — UIT commissioning

**Depends on:** T01–T12. **Deliverable:** exact environment lock and matching actual-server preflight/smoke reports.  
**Location:** UIT Ubuntu/Slurm; GPU workloads only on allocated compute nodes.

- [ ] T13.1 Inspect server paths, scheduler settings and actual H200 MIG profile read-only; record verified operational values.
- [ ] T13.2 Prepare/verify the environment, exact pretrained snapshot and licensed/reviewed non-speech noise; save `requirements/uit-lock.txt` and asset identities.
- [ ] T13.3 Audit server Train and token lengths: exact hash, 1,545 sources, 5,036 utterances and zero unresolved input errors.
- [ ] T13.4 Run GPU preflight and isolated full-model smoke through Slurm; test warning signal delivery and measure memory/window/checkpoint times.
- [ ] T13.5 Verify the actual 192/3,932,160 LoRA assertions, dtype/kernel behavior and resumed-state agreement; capture report hashes.
- [ ] T13.6 Mark readiness only if all protected identities match and every required check passes; otherwise record blockers without changing locked settings.

### T14 — Official training and final artifacts

**Depends on:** T01–T13 and operator scheduling. **Deliverable:** completed official run, final FP16 model and validated Train transcripts.  
**Location:** Slurm GPU allocations; potentially multiple allocations continuing the same run.

- [ ] T14.1 Verify admission evidence, reserve a unique official run ID and initialize the new seed-42 adapter from the exact pretrained base.
- [ ] T14.2 Submit scheduled training and resume interrupted attempts from complete compatible checkpoints, preserving immutable prior artifacts.
- [ ] T14.3 Verify exactly 10 epochs and 3,150 successful optimizer updates for the confirmed dataset, with all required checkpoints and no lost occurrences.
- [ ] T14.4 Merge/export the final epoch-10 adapter; verify the actual full FP16 artifact, offline reload, numerical report and hashes.
- [ ] T14.5 Infer original Train audio and validate all 5,036 utterance records and 1,545 source records with exact current/history mapping.
- [ ] T14.6 Finalize run/output validation manifests and summaries; mark end-to-end completion only when all acceptance criteria pass.

## Gate Checklist

- [ ] G1 — CPU implementation/tests, CLI behavior and Ubuntu script syntax pass.
- [ ] G2 — Server dataset, labels, snapshot, noise and dependency identities are verified.
- [ ] G3 — Actual GPU preflight, smoke/resume, memory and Slurm signal tests pass for matching inputs.
- [ ] G4 — Official training completes exactly 10 epochs and the audited successful update budget.
- [ ] G5 — Actual final merged FP16 model passes export, numerical and offline reload checks.
- [ ] G6 — Complete utterance/source transcripts and run artifacts pass output validation.

## Current Blockers and Evidence Boundaries

| ID | Condition | Current evidence | Required resolution |
|---|---|---|---|
| B01 | UIT access and Slurm resource settings | Not verified in this workspace | T13.1: record allowed account/partition/QoS/GRES/time limits and allocation behavior |
| B02 | Exact pretrained snapshot on UIT | Not verified | T13.2: verify revision, complete files, architecture and hashes |
| B03 | Valid external non-speech noise | Not verified | T13.2: establish licensing/review evidence, hashes, ordered manifest and audio validity |
| B04 | Compatible exact environment and GPU kernels | Candidate versions only | T13.2/T13.4: lock actual stack and pass runtime probes |
| B05 | Full Train token-length validity | Local WAV/hash review is not tokenizer evidence | T13.3: tokenize all references with the exact snapshot, without truncation |
| B06 | Memory fit, correct signals and full-model resume | Not measured on the MIG instance | T13.4/T13.5: actual allocation tests and recorded comparisons |

“Not verified” does not mean “confirmed missing.” These are barriers to official execution, not reasons to skip writing or CPU-testing implementation.

## Execution Record

No implementation or server-operation task has been executed yet. When work begins, append one row per verified task/attempt using:

| Task or gate | Status | Environment / run / attempt | Command and evidence path/hash | Blocker or next action |
|---|---|---|---|---|
| T01–T14, G1–G6 | Not started | No execution environment or run created | No execution evidence yet | Implement in dependency order; preserve locked configuration |

Allowed tracker statuses are `not_started`, `in_progress`, `blocked` and `complete`. Mark a task complete only when its own deliverable is verified. Software construction can be complete while the separate operational tasks T13/T14 remain pending; the overall system is not accepted until G6.
