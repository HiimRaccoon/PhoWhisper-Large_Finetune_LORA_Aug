from dataclasses import replace
import json
import pytest


def identity_fixture():
    from asr.contracts import ArtifactIdentity
    return ArtifactIdentity(method_hash='method',dataset_hash='dataset',manifest_hash='manifest',
                            snapshot_hash='snapshot',noise_hash='noise',runtime_hash='runtime',dependency_hash='deps')


def training_fixture(config,k=16):
    from training_helpers import ToyLoss,examples,bundle_for
    from asr.checkpoint import TrainingObjects,capture_training
    from asr.training import compute_budget,create_optimizer,run_window,epoch_order
    from asr.schedule import create_scheduler
    from asr.contracts import TrainingProgress
    from asr.config import canonical_hash
    bundle=bundle_for(ToyLoss()); optimizer=create_optimizer(bundle,config)
    scheduler=create_scheduler(optimizer,compute_budget(32))
    order=epoch_order(32,0)
    progress=TrainingProgress(permutation_hash=canonical_hash(order))
    run_window(bundle,optimizer,scheduler,None,examples(k),progress)
    objects=TrainingObjects(bundle,optimizer,scheduler,None)
    return objects,capture_training(objects,progress,identity_fixture(),order,config=config)


def test_checkpoint_is_immutable_and_hash_corruption_fails(local_config):
    from asr.checkpoint import save_checkpoint,load_checkpoint
    objects,state=training_fixture(local_config)
    run=local_config.paths.project_root/'runs/r'
    path=save_checkpoint(run,'attempt1',state)
    loaded=load_checkpoint(path,identity_fixture())
    assert loaded.progress.global_update==1
    assert loaded.progress.next_position==16
    with pytest.raises(FileExistsError): save_checkpoint(run,'attempt1',state)
    with pytest.raises(ValueError,match='identity'):
        load_checkpoint(path,replace(identity_fixture(),noise_hash='changed'))
    with (path/'state.pt').open('ab') as f: f.write(b'corruption')
    with pytest.raises(ValueError,match='hash'):
        load_checkpoint(path,identity_fixture())


def test_pending_or_incomplete_checkpoints_are_not_resumable(local_config):
    from asr.checkpoint import save_checkpoint,load_checkpoint,discover_checkpoints
    objects,state=training_fixture(local_config)
    state.progress.pending_micro_steps=1
    with pytest.raises(ValueError,match='pending'):
        save_checkpoint(local_config.paths.project_root/'runs/r','a',state)
    incomplete=local_config.paths.project_root/'runs/r/checkpoints/a/.tmp-checkpoint-cut'
    incomplete.mkdir(parents=True)
    (incomplete/'state.pt').write_bytes(b'partial')
    assert discover_checkpoints(local_config.paths.project_root/'runs/r')==[]
    with pytest.raises(ValueError,match='incomplete'):
        load_checkpoint(incomplete,identity_fixture())


def test_checkpoint_writer_excludes_concurrent_attempt_owner(local_config):
    from filelock import FileLock
    from asr.checkpoint import save_checkpoint
    objects,state=training_fixture(local_config)
    attempt=local_config.paths.project_root/'runs/r/checkpoints/a'; attempt.mkdir(parents=True)
    with FileLock(str(attempt/'.writer.lock'),timeout=0):
        with pytest.raises(RuntimeError,match='writer'):
            save_checkpoint(local_config.paths.project_root/'runs/r','a',state)


def test_interruption_at_rename_never_publishes_staging_checkpoint(local_config,monkeypatch):
    import os
    from asr.checkpoint import save_checkpoint,discover_checkpoints
    objects,state=training_fixture(local_config)
    run=local_config.paths.project_root/'runs/r'
    def interrupted(*args): raise OSError('simulated termination before directory publication')
    monkeypatch.setattr(os,'rename',interrupted)
    with pytest.raises(OSError,match='simulated'):
        save_checkpoint(run,'cut',state)
    assert discover_checkpoints(run)==[]
