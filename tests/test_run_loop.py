from dataclasses import replace
import threading
from training_helpers import ToyLoss,examples,bundle_for


def objects_fixture(config,n):
    from asr.checkpoint import TrainingObjects
    from asr.training import compute_budget,create_optimizer
    from asr.schedule import create_scheduler
    bundle=bundle_for(ToyLoss()); opt=create_optimizer(bundle,config)
    return TrainingObjects(bundle,opt,create_scheduler(opt,compute_budget(n)),None)


def test_loop_traverses_ten_epochs_and_combines_final_checkpoint_events(local_config):
    from asr.training import train_loop
    from asr.checkpoint import discover_checkpoints,load_checkpoint
    from test_checkpoint import identity_fixture
    identity=replace(identity_fixture(),method_hash=local_config.method_hash)
    objects=objects_fixture(local_config,80); visits=[]; events=[]
    def prepare(row,epoch):
        visits.append((epoch,row)); return examples(1)[0]
    run=local_config.paths.project_root/'runs/loop'
    progress=train_loop(objects,list(range(80)),prepare,local_config,run,'a',identity,event_callback=events.append)
    assert progress.completed_epochs==10 and progress.epoch==10
    assert progress.global_update==50 and progress.global_micro_step==800
    assert len(visits)==len(set(visits))==800
    assert progress.successful_updates_by_epoch==[5]*10
    final=load_checkpoint(discover_checkpoints(run)[-1],identity)
    assert final.metadata['kind']=='terminal'
    assert {'interval','epoch','terminal'}.issubset(final.metadata['reasons'])


def test_loop_resumes_after_a_boundary_stop_without_losing_or_repeating_rows(local_config):
    from asr.training import train_loop
    from asr.checkpoint import discover_checkpoints,load_checkpoint,restore_training
    from test_checkpoint import identity_fixture
    identity=replace(identity_fixture(),method_hash=local_config.method_hash)
    run=local_config.paths.project_root/'runs/resumed'; visits=[]; flag=threading.Event()
    def prepare(row,epoch): visits.append((epoch,row)); return examples(1)[0]
    def event(record):
        if record['event']=='optimizer_update' and record['global_update']==1: flag.set()
    initial=objects_fixture(local_config,17)
    interrupted=train_loop(initial,list(range(17)),prepare,local_config,run,'first',identity,
                           event_callback=event,stop_requested=flag.is_set)
    assert interrupted.completed_epochs==0 and interrupted.next_position==16
    state=load_checkpoint(discover_checkpoints(run)[-1],identity)
    resumed=objects_fixture(local_config,17); progress=restore_training(state,resumed)
    final=train_loop(resumed,list(range(17)),prepare,local_config,run,'second',identity,
                     progress=progress,permutation=state.permutation)
    assert final.completed_epochs==10 and final.global_update==20
    assert len(visits)==len(set(visits))==170
