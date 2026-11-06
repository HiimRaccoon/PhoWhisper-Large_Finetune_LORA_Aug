# PhoWhisper-large LoRA ASR Task Tracker

**Specification:** [specs.md](specs.md).  
**Implementation plan:** [plan.md](plan.md).  
**Status:** T01–T12 implementation in progress; actual UIT/GPU and official execution remain pending.  
**Tracking rule:** this file owns completion status. Check an item only after its acceptance evidence exists; record commands, report paths/hashes and blockers in the execution record. A written test or a skipped GPU test is not a passing test.

## Scope and Execution Rules

The final locked configuration in specification section 5 applies to every task. Do not change the base/revision, encoder + decoder LoRA, seed, dataset, augmentation, 10 epochs, batch 1 / accumulation 16, optimizer/schedule, checkpoint selection or FP16 export.

Keep source data read-only and external to the Ubuntu project. Never access `CS406/`. Do not use Public/Private Test, environment state, tool labels or assistant turns for ASR training. Preserve every duplicate occurrence, fail on conflicts, and never silently skip invalid rows.

T01–T12 build and test software. T01.7 adds a nonblocking early feasibility record, not a new phase. T13 commissions the actual UIT environment and has a CPU-only entry for deferred Linux checks. T14 operates the official run only after matching preflight/smoke evidence and operator scheduling. This planning revision has performed none of those tasks.

For each implementation task, follow the plan's failing-test → minimal implementation → passing-test steps. Record evidence, then commit the verified task when executing within an existing Git workflow. Do not initialize a repository or submit jobs merely to satisfy this tracker.

## Subtask Acceptance and Evidence

Each item below is an executable subtask within its existing main task. Its objective is stated on the checklist line; affected files, dependencies, exact verification and measurable acceptance/evidence are defined in the corresponding task/subtask of [plan.md](plan.md). Task-level fields are inherited unless a subtask narrows them. For the refined T03/T05/T06/T07/T08/T10/T11 components, match the same subtask ID in the plan. Attach the command/result, environment and artifact/report identities before checking an item; implement behavioral tests before the software they exercise.

Portable CPU evidence, actual Linux evidence, real-snapshot evidence and allocated-GPU evidence are distinct. T12 software readiness can precede G1.linux when WSL/Linux is unavailable. Only CPU-only T13.1 may then start; full G1 and G2 must pass before T13.4/G3. Early unknown/unavailable feasibility fields never count as commissioning success.

## Ordered Work

### T01 — Locked configuration and package contract

**Depends on:** none. **Deliverable:** package/configuration schema, unchanged official YAML, canonical method identity, Git artifact isolation and an early feasibility record.  
**Files:** `.gitignore`, `pyproject.toml`, `configs/official.yaml`, package configuration/contracts, `tests/{test_config,test_gitignore}.py`; generated feasibility evidence stays under `manifests/`.  
**Verify:** `pytest tests/test_config.py tests/test_gitignore.py -q`; T01.7 uses a documented feasibility review, not a GPU test.

- [x] T01.1 Write rejection tests for every locked-field override, invalid types/keys and unsafe operational paths.
- [x] T01.2 Run the tests and record the intended failures before implementation.
- [x] T01.3 Implement `OfficialConfig`, `load_config`, `validate_locked`, project packaging and candidate requirements.
- [x] T01.4 Verify official epochs 10, scope `encoder_decoder`, deduplication false, and stable method hashes across permitted path changes.
- [x] T01.5 Record passing configuration/package evidence and checkpoint that change in the existing Git workflow when applicable.
- [x] T01.6 Implement/test `.gitignore` using a tiny isolated Git fixture: generated weights/snapshots/checkpoints/exports/runs/transcripts/datasets/WAVs/venvs/caches/temp checkpoints/local secret-bearing config stay untracked; source/templates/tests/docs stay trackable. Record `git check-ignore --no-index`, fixture staged-path assertions and a read-only existing-tracked-path audit.
- [x] T01.7 Record early UIT path/resource/MIG/snapshot/noise/dependency/memory feasibility from an operator report or authorized read-only/CPU inspection when access exists. Otherwise record access unavailable and unresolved fields without blocking local work; review before T05/T09 GPU integration, and leave actual acceptance to T13.

### T02 — Safe projection and Train audit

**Depends on:** T01. **Deliverable:** read-only audit, stable utterance manifest, error and duplicate/conflict reports.  
**Verify:** `pytest tests/test_projection.py tests/test_audio_audit.py tests/test_paths.py -q`.

- [x] T02.1 Write fixtures covering current/history index mapping, assistant exclusion, truncated/invalid WAVs, missing files, duplicate/conflicting PCM and symlink/path escapes.
- [x] T02.2 Confirm the new tests fail for the intended missing behavior.
- [x] T02.3 Implement safe resolution, PCM decoding/hashing, transcript normalization, source/utterance identity checks and audit outputs.
- [x] T02.4 Verify duplicate retention, conflict failure, original order/index preservation, zero dataset writes and no environment/label exports.
- [x] T02.5 Record passing evidence and commit when applicable.

### T03 — Asset verification and processor

**Depends on:** T01–T02. **Deliverable:** independently verified snapshot/noise identities and explicit Whisper feature/mask/label contracts.  
**Files:** `src/asr/{assets,processor}.py`, `tests/{test_assets,test_processor}.py`, `tests/gpu/test_snapshot_mask_forward.py`.  
**Verify:** `pytest tests/test_assets.py tests/test_processor.py -m "not gpu and not snapshot" -q`; actual snapshot forward is deferred to T13.

- [x] T03.1 Verify the exact local snapshot with CPU hash/completeness/architecture/containment fixtures; record passing tests and snapshot identity without claiming real-snapshot readiness.
- [x] T03.2 Verify noise ordering, licensing/review/non-speech metadata, PCM format/power and hashes; record fixture results and ordered manifest identities.
- [x] T03.3 Implement explicit `return_attention_mask=True`, no-truncation 480,000-sample padding and features `[1, 80, 3000]` with integer frame mask `[1, 3000]`. CPU tests for lengths 16000/16001/480000 require mask sums 100/101/3000, mask semantics and preservation of the final input sample. Implement the separately marked actual-snapshot mask-forward test for T13.
- [x] T03.4 Verify unchanged label rules: NFC + trim, decoder-start handling, 447 accepted / 448 rejected, EOS retention and padding-only `-100` even with shared PAD/EOS IDs; record boundary assertions.
- [x] T03.5 Integrate asset/processor records and pass the CPU suite; record evidence and pending real-tokenizer/forward acceptance in T13 before committing when applicable.

### T04 — Deterministic E4 augmentation

**Depends on:** T02–T03. **Deliverable:** one deterministic waveform view per utterance/epoch, with complete decisions/fallback logs.  
**Verify:** `pytest tests/test_augmentation.py -q`.

- [x] T04.1 Write independent seed/draw-trace fixtures and tests for all gates, speed/volume/noise transforms, noise wrapping, zero power and fallback boundaries.
- [x] T04.2 Confirm failing tests before implementation.
- [x] T04.3 Implement the exact SHA-256 seed, conditional draw order, interpolation convention, SNR formula and clean fallback behavior.
- [x] T04.4 Verify seed `18138155238159038922` for `42:0:train_0001/current:augmentation-v1`, replay waveform equality, unchanged references and no redrawing/dataset doubling.
- [x] T04.5 Record golden fixture provenance, NumPy/noise identities and passing evidence; commit when applicable.

### T05 — Exact LoRA and runtime precision

**Depends on:** T01, T03. **Deliverable:** exact targets, generic PEFT initialization, safe precision profile and checkpointed gradient flow.  
**Files:** `src/asr/{model,precision}.py`, `tests/{test_lora_targets,test_precision}.py`.  
**Verify:** `pytest tests/test_lora_targets.py tests/test_precision.py -m "not gpu" -q`; real counts/kernel probes remain T13 evidence.

- [x] T05.1 Resolve the exact six module families; fixture tests reject missing/extra/out-of-scope targets and assert the expected set of 192 names.
- [x] T05.2 Build fresh seed-42 generic PEFT/Whisper initialization with only A/B trainable, exact locked LoRA settings and no warm start/quantization; record wrapper/scope fixture tests, with the actual 3,932,160 count pending UIT.
- [x] T05.3 Implement allocation-before-CUDA guards and the existing FP32 storage / BF16-or-FP16 autocast strategy; record mocked CPU profile tests and review T01.7's nonblocking feasibility record.
- [x] T05.4 Verify non-reentrant checkpointed encoder/decoder gradient flow and frozen base state on a tiny CPU fixture; initial zero A gradients are allowed, with A connectivity tested after B changes at nonzero LR.
- [x] T05.5 Integrate `TrainingBundle` and runtime metadata, pass all CPU tests and record the exact actual-GPU checks still pending T13; commit when applicable.

### T06 — Training windows and scheduler

**Depends on:** T01–T05. **Deliverable:** consistent budget/order, explicit scheduler indexing, A/B-only AdamW and correct successful-window progress.  
**Files:** `src/asr/{training,schedule}.py`, shared progress records, `tests/{test_accumulation,test_schedule,test_training_guards}.py`.  
**Verify:** `pytest tests/test_accumulation.py tests/test_schedule.py tests/test_training_guards.py -q`.

- [x] T06.1 Verify 5,036 rows produce `315 / 3150 / 158` budget values, deterministic epoch permutations and 314 windows of 16 plus one of 12; record order/coverage evidence.
- [x] T06.2 Implement the documented zero-based successful-call LR timeline and round-trip scheduler/optimizer state. Test actual LR at `u=0,1,157,158,159,3149` and post-final state, plus uninterrupted/resumed remaining traces around warmup and the final call.
- [x] T06.3 Verify exact A/B-only AdamW settings and that finite non-skipped LR-0 calls advance moments/step counters and budget while leaving parameters unchanged. Successful/scaler-skipped calls can both return `None`; tests must distinguish actual invocation.
- [x] T06.4 Verify loss/k accumulation for windows 16 and 12, fixed scale within the window and unscale/clip/optimizer/scheduler order against per-utterance mean-loss references.
- [x] T06.5 Verify clean progress/terminal counters and hard failure on OOM/non-finite/scaler skips without scheduler/completed-progress advancement; save passing suite/traces and commit when applicable.

### T07 — Checkpoints and resume

**Depends on:** T01, T04–T06. **Deliverable:** separately tested immutable publication, complete serialization, identity validation and exact continuation.  
**Files:** `src/asr/checkpoint.py`, `tests/{test_checkpoint,test_resume}.py`.  
**Verify:** `pytest tests/test_checkpoint.py tests/test_resume.py -q`; actual POSIX/filesystem tests also feed G1.linux/T13.1.

- [x] T07.1 Test/implement atomic publication and writer exclusion with interrupted writes, collisions and concurrent attempts; record that incomplete checkpoints are never resumable.
- [x] T07.2 Round-trip adapter/optimizer/scheduler/scaler/RNG/permutation/progress and terminal identity; reject pending gradients or `pending_micro_steps != 0` and save field/hash comparisons.
- [x] T07.3 Validate every protected identity before continuation, reconstruct objects then restore RNG last, and assert the exact next LR/cursor; record mismatch and RNG diagnostics.
- [x] T07.4 Compare uninterrupted/resumed sample IDs, augmentation, LR, adapter/optimizer state and budget across mid-epoch, partial/epoch boundaries, warmup and final-call restore; record passing suite/traces and commit when applicable.

### T08 — Lifecycle, admission, logs and CLI

**Depends on:** T01–T07. **Deliverable:** independently verified lifecycle, structured logs, admission, orchestration and CLI integration.  
**Files:** `src/asr/{artifacts,logging,preflight,cli,training,__main__}.py` and `tests/{test_artifacts,test_logging,test_preflight,test_run_loop,test_cli}.py`.  
**Verify:** `pytest tests/test_artifacts.py tests/test_logging.py tests/test_preflight.py tests/test_cli.py tests/test_run_loop.py -q`.

- [x] T08.1 Implement/test run identity, unique attempts, ownership and allowed lifecycle transitions; record duplicate/conflicting-write rejection and immutable prior artifacts.
- [x] T08.2 Implement/test all required training/augmentation JSONL fields and fallback summaries; record parseable fixture logs with attributable actual LR/progress/failure events.
- [x] T08.3 Implement/test CPU/GPU preflight and identity-bound admission by composing existing validators; stale/missing/tampered reports and fabricated GPU success must fail before training.
- [x] T08.4 Implement/test `execute_run` over 10 tiny-model epochs, resume cursors, stop-at-window behavior and update-50/epoch/terminal checkpoint triggers; record complete visitation and terminal adapter linkage.
- [x] T08.5 Wire audit/preflight/train/resume and nonzero diagnostic exits; pass the integration suite and prove no CLI training path bypasses admission or confuses fresh initialization with resume; commit when applicable.

### T09 — Isolated full-model smoke

**Depends on:** T01–T08. **Deliverable:** smoke harness and actual-model continuation report, separate from the official run.  
**Verify:** `pytest tests/test_smoke.py -q`; later `pytest tests/gpu/test_full_model_smoke.py -m gpu -q` inside an allocation.

- [ ] T09.1 Write harness and real-model tests comparing uninterrupted/resumed execution, frozen base state and adapter gradient connectivity.
- [ ] T09.2 Confirm CPU harness tests fail; leave GPU execution pending until allocation exists.
- [ ] T09.3 Implement isolated windows 16, 16, 12, checkpoint/resume, the 30-second/447-label shape probe and runtime/memory recording.
- [ ] T09.4 Verify CPU tests and implement the GPU assertions for matching IDs/augmentation/LR/progress and adapter state within the declared tolerance; leave actual GPU execution to T13. No smoke weights may initialize official training.
- [ ] T09.5 Record passing CPU evidence and the ready-to-run GPU test/harness; commit code when applicable. A passing actual-server `SmokeReport` remains pending T13/G3.

### T10 — Final merge and FP16 export

**Depends on:** T03, T05, T07–T08. **Deliverable:** semantic terminal-checkpoint validation and final-only export/reload.  
**Files:** `src/asr/{export,cli}.py`, `tests/test_export.py`, `tests/gpu/test_export_reload.py`.  
**Verify:** `pytest tests/test_export.py -m "not gpu" -q`; actual final-model checks remain T14/G5.

- [ ] T10.1 Accept both canonical terminal epoch 10 and fully completed last-index-9 representations only with `completed_epochs==10`, all 3,150 successful calls/50,360 micro-steps, zero pending accumulation and correct terminal adapter identity. Reject ordinary index-9 checkpoints, 3,149 calls, pending gradients, incomplete visits and falsely named final paths; save the acceptance matrix.
- [ ] T10.2 Bind exact base/revision/run/adapter identities and reject existing/conflicting export destinations before merge; record failure diagnostics.
- [ ] T10.3 Merge FP32 base/adapter in eval mode, export complete finite FP16 weights plus processor/tokenizer/generation/provenance, and pass tiny-model publication/metadata tests.
- [ ] T10.4 Verify fresh-process local-only reload without PEFT and declared logit tolerances; record CPU mechanics and leave real final-model certification pending T14.
- [ ] T10.5 Wire `merge-export` through terminal/identity validation before loading/publishing, pass CPU tests and record evidence/commit when applicable.

### T11 — Train inference and output validation

**Depends on:** T02–T03, T08, T10. **Deliverable:** locked generation, durable interruption recovery, source grouping and validated public transcript files.  
**Files:** `src/asr/{inference,outputs,cli}.py`, `tests/{test_inference,test_inference_resume,test_outputs}.py`.  
**Verify:** `pytest tests/test_inference.py tests/test_inference_resume.py tests/test_outputs.py -q`; actual filesystem locking is verified in T13.1.

- [ ] T11.1 Implement/test offline eval/inference-mode greedy generation and mask forwarding, preserving all locked settings and raw token/EOS-limit evidence; record resolved-generation assertions.
- [ ] T11.2 Implement/test a one-writer ordered JSONL prefix journal bound to model/config/manifest/generation identities; crash-injection tests cover newline/fsync interruptions, trailing fragments, corrupt/duplicate/out-of-order records, mismatches and repeated restart without duplicate committed output.
- [ ] T11.3 Implement/test original-order grouping for current and user-history turns, including empty/interleaved histories; record exact expected source records using the unchanged output schema.
- [ ] T11.4 Test completeness before publication and interruption between candidate validation, public file installs and the final completion report; require matching hashes/identities, no missing/extra/empty/incomplete predictions and no reference fallback.
- [ ] T11.5 Wire compatible continuation into repeated `infer-train`, integrate `validate-output` and publish `transcripts_complete` only after valid completion evidence; save CPU suite results and commit when applicable.

### T12 — Setup, Slurm scripts and README

**Depends on:** T01, T05, T07–T11 software readiness, not prior T13 access. **Deliverable:** configurable deployment scripts, correct signals and complete operational documentation.  
**Verify:** portable fake-scheduler tests now; the per-file Bash syntax loop and `pytest tests/test_slurm_scripts.py tests/test_inference_resume.py tests/test_checkpoint.py -m linux -q` on WSL/Linux when available, or CPU-only T13.1. Record actual executed tests and hashes.

- [ ] T12.1 Write fake-scheduler tests for required resources, argument quoting, logging, exit status, repeated job IDs and signal delivery.
- [ ] T12.2 Confirm failing tests.
- [ ] T12.3 Implement setup/submission wrappers and preflight/smoke/train/export/infer job scripts, with all generated paths under the project.
- [ ] T12.4 Verify portable wrapper/resource/guard behavior and document all lifecycle/continuation commands. Execute actual Bash/POSIX signal/lock/publication checks on available WSL/Linux, or explicitly defer G1.linux to CPU-only T13.1; a Windows mock/skip is not Linux evidence.
- [ ] T12.5 Record portable CPU evidence, real Linux results or their explicit deferral, and commit software when applicable; actual Linux verification remains mandatory before any GPU/official stage, and scripts never auto-submit official training.

### T13 — UIT commissioning

**Depends on:** T01–T12 software readiness and G1.cpu; G1.linux may be pending only at CPU-only T13.1 entry. **Deliverable:** Linux/project-filesystem verification, exact environment lock and matching actual-server preflight/smoke reports.  
**Location:** UIT Ubuntu/Slurm; GPU workloads only on allocated compute nodes.

- [ ] T13.1 Enter CPU-only commissioning, revisit the early feasibility record and inspect allowed UIT paths/resource configuration without CUDA/GPU model access. Prepare CPU-test dependencies as needed; close deferred G1.linux with actual Bash/POSIX tests and verify project-filesystem lock/flush/rename behavior even if WSL checks passed. Record hashes and actual executed counts.
- [ ] T13.2 Prepare/verify the complete environment, exact pretrained snapshot and licensed/reviewed non-speech noise; save `requirements/uit-lock.txt` and asset identities. CPU dependency preparation for T13.1 needs no prior GPU/asset gate.
- [ ] T13.3 Audit server Train and token lengths: exact hash, 1,545 sources, 5,036 utterances and zero unresolved input errors.
- [ ] T13.4 After full G1 and G2 pass, run GPU preflight, T03's real-snapshot mask-forward compatibility test and isolated full-model smoke through Slurm; verify allocated MIG/driver facts, warning signal delivery and memory/window/checkpoint times.
- [ ] T13.5 Verify the actual 192/3,932,160 LoRA assertions, dtype/kernel behavior and resumed-state agreement; capture report hashes.
- [ ] T13.6 Mark readiness only if all protected identities match and every required check passes; otherwise record blockers without changing locked settings.

### T14 — Official training and final artifacts

**Depends on:** T01–T13 and operator scheduling. **Deliverable:** completed official run, final FP16 model and validated Train transcripts.  
**Location:** Slurm GPU allocations; potentially multiple allocations continuing the same run.

- [ ] T14.1 Verify admission evidence, reserve a unique official run ID and initialize the new seed-42 adapter from the exact pretrained base.
- [ ] T14.2 Submit scheduled training and resume interrupted attempts from complete compatible checkpoints, preserving immutable prior artifacts.
- [ ] T14.3 Verify exactly 10 epochs and 3,150 successful optimizer updates for the confirmed dataset, with all required checkpoints and no lost occurrences.
- [ ] T14.4 Merge/export the final epoch-10 adapter; verify the actual full FP16 artifact, offline reload, numerical report and hashes.
- [ ] T14.5 Infer original Train audio, continue only a matching durable journal after interruption and validate all 5,036 utterance / 1,545 source records with exact mapping and a completion report matching the published file hashes.
- [ ] T14.6 Finalize run/output validation manifests and summaries; mark end-to-end completion only when all acceptance criteria pass.

## Gate Checklist

- [ ] G1 — Both G1.cpu (portable CPU tests/CLI/config/Git isolation) and G1.linux (actual Linux Bash/POSIX checks) pass. G1.cpu alone allows CPU-only T13.1, not GPU/official execution.
- [ ] G2 — Server dataset, labels, snapshot, noise and dependency identities are verified.
- [ ] G3 — Actual GPU preflight, real-snapshot feature/mask forward, smoke/resume, memory and Slurm signal tests pass for matching inputs.
- [ ] G4 — Official training proves 10 completed epochs, the audited successful-call budget, zero pending accumulation and the correct terminal adapter/checkpoint identity.
- [ ] G5 — Actual final merged FP16 model passes export, numerical and offline reload checks.
- [ ] G6 — Complete utterance/source transcripts, durable continuation/publication and run artifacts pass output validation with a valid matching completion report.

## Current Blockers and Evidence Boundaries

| ID | Condition | Current evidence | Required resolution |
|---|---|---|---|
| B01 | UIT access and Slurm resource settings | Not verified; Windows access is not assumed | T01.7 records early feasibility/unknowns; T13.1 verifies actual allowed settings, T13.4 verifies allocated GPU facts |
| B02 | Exact pretrained snapshot on UIT | Not verified | T13.2: verify revision, complete files, architecture and hashes |
| B03 | Valid external non-speech noise | Not verified | T13.2: establish licensing/review evidence, hashes, ordered manifest and audio validity |
| B04 | Compatible exact environment and GPU kernels | Candidate versions only | T13.2/T13.4: lock actual stack and pass runtime probes |
| B05 | Full Train token-length validity | Local WAV/hash review is not tokenizer evidence | T13.3: tokenize all references with the exact snapshot, without truncation |
| B06 | Memory fit, correct signals and full-model resume | Not measured on the MIG instance | Early risks recorded in T01.7; acceptance requires T13.4/T13.5 actual allocation tests |
| B07 | Actual Linux/script and project-filesystem semantics | Not verified; no local WSL/Linux availability assumed | T12 documents/runs G1.linux when possible; CPU-only T13.1 closes pending checks and verifies actual project locking/flush/rename behavior |

“Not verified” does not mean “confirmed missing.” These are barriers to official execution, not reasons to skip writing or CPU-testing implementation.

## Platform Evidence Routing

| Evidence | Where it can be produced | Completion boundary |
|---|---|---|
| Early feasibility | Operator report or authorized read-only/CPU inspection; access-unavailable record permitted | T01.7 records status only; never satisfies G2/G3 |
| Portable CPU software tests | Local supported environment, including Windows | G1.cpu; software tasks can complete with real-platform tests pending |
| Actual Linux syntax/POSIX tests | WSL/Linux if available, otherwise CPU-only UIT T13.1 | G1.linux before GPU stages; actual UIT filesystem behavior is verified in T13.1 |
| Real snapshot feature/mask forward and full-model smoke | UIT compute allocation with actual pinned assets | T13/G3; no fixture/mock substitutes |
| Final model and complete transcripts | Official run/export/inference allocations | T14/G4–G6; no smoke/tiny-model substitutes |

G1.cpu command: `pytest tests -m "not gpu and not snapshot and not linux" -q`. G1.linux requires the exact T12 per-file Bash loop and Linux-marked tests, with no required test skipped. These are evidence components of existing G1, not new gates.

## Execution Record

Software completion below excludes actual Linux/snapshot/GPU acceptance unless separately recorded.

| Task | Status | Local evidence | Remaining operational evidence |
|---|---|---|---|
| T01 | complete | pytest tests/test_config.py tests/test_gitignore.py -q: 11 passed; RED config/ignore absence observed; real Git fixture passed; feasibility unavailable recorded | T13 actual snapshot/GPU/Linux commissioning; T14 official execution |
| T02 | complete | RED projection/audit/path modules absent; GREEN 10 task tests and full suite 21 passed, 1 Linux test deselected. Read-only actual Train audit: valid, 1545 sources, 5036 utterances, zero duplicate/conflict/errors; manifests/train/audit.json. | T13 actual snapshot/GPU/Linux commissioning; T14 official execution |
| T03 | complete | RED asset/processor modules absent; GREEN 8 task tests; full portable suite 29 passed, 2 deselected. Real WhisperFeatureExtractor verifies 3000-frame masks at 16000/16001/480000 samples; label 447/448 and shared EOS/PAD covered. Real-snapshot GPU forward exists and is pending T13. | T13 actual snapshot/GPU/Linux commissioning; T14 official execution |
| T04 | complete | RED augmentation module absent; GREEN 9 deterministic/speed/volume/noise/fallback tests. Full suite 38 passed, 2 deselected. Golden seed and zero-waveform fixture independently derived; exact gate draw order tested through RNG boundary fixtures. | T13 actual snapshot/GPU/Linux commissioning; T14 official execution |
| T05 | complete | RED model/precision modules absent; GREEN 7 task tests; full suite 45 passed, 2 deselected. Real tiny Whisper+PEFT forwards/backwards prove encoder/decoder gradients and frozen base; meta-device full topology proves 192 exact targets. Real H200 counts/kernel/memory remain pending T13. | T13 actual snapshot/GPU/Linux commissioning; T14 official execution |
| T06 | complete | RED training/schedule modules absent; GREEN 13 task tests, full suite 58 passed, 2 deselected. Exact LR boundaries and resumed traces, 16/12 mean-loss gradients, LR-zero AdamW moments, real CPU GradScaler success, scaler-skip/nonfinite/OOM guards verified. | T13 actual snapshot/GPU/Linux commissioning; T14 official execution |
| T07 | complete | RED checkpoint module absent; GREEN 5 checkpoint/resume tests. Full suite 64 passed, 2 deselected. Immutable atomic publication, rename interruption, writer exclusion, hashes, safe tensor-only state loading, RNG-last and optimizer/scheduler/cursor restore verified. Regression RED/GREEN also prevents outputs inside a local source-dataset root. | T13 actual snapshot/GPU/Linux commissioning; T14 official execution |
| T08 | complete | RED artifacts/logging/preflight/loop/CLI seams absent; GREEN 10 task tests; full suite 74 passed, 2 deselected. Tiny-model 10-epoch traversal, combined interval/terminal events, mid-epoch resume, logging totals, stale evidence rejection and CLI GPU-allocation admission verified. Actual Linux/GPU checks pending. | T13 actual snapshot/GPU/Linux commissioning; T14 official execution |

T13, T14 and G1–G6 remain pending. No full training job has been submitted.
