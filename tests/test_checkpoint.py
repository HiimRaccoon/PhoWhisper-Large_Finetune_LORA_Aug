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
    assert loaded.metadata['optimizer_parameter_names'] == [['lora_A']]
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


@pytest.mark.linux
def test_posix_kill_during_checkpoint_publication_releases_writer(local_config):
    import os,subprocess,sys,time
    from pathlib import Path
    assert sys.platform.startswith('linux'),'G1.linux requires actual Linux'
    from asr.checkpoint import save_checkpoint,discover_checkpoints,load_checkpoint
    project=local_config.paths.project_root; root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'src'),str(root/'tests')]))
    code='''
import os,sys,time
from pathlib import Path
from asr.config import load_config
from asr.checkpoint import save_checkpoint
from test_checkpoint import training_fixture
project=Path(sys.argv[1]); config=load_config(Path(sys.argv[2])/'configs/official.yaml').with_paths(project_root=project,data_root=project/'dataset',snapshot=project/'assets/pretrained',noise_manifest=project/'assets/noise/manifest.json')
_,state=training_fixture(config)
def stop(*_):
    (project/'ready').write_text('ready')
    while True: time.sleep(.02)
os.rename=stop
save_checkpoint(project/'runs/r','cut',state)
'''
    process=subprocess.Popen([sys.executable,'-c',code,str(project),str(root)],env=env)
    try:
        deadline=time.monotonic()+15
        while not (project/'ready').exists() and time.monotonic()<deadline: time.sleep(.02)
        assert (project/'ready').is_file()
        process.kill(); process.wait(timeout=10)
        assert discover_checkpoints(project/'runs/r')==[]
        _,state=training_fixture(local_config)
        path=save_checkpoint(project/'runs/r','cut',state)
        assert load_checkpoint(path,state.identity).progress.global_update==state.progress.global_update
    finally:
        if process.poll() is None: process.kill(); process.wait()
