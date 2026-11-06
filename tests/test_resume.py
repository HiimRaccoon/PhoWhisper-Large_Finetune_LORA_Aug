import copy
import random
import numpy as np
import torch


def test_resume_restores_model_moments_rng_cursor_and_next_lr(local_config):
    from test_checkpoint import training_fixture,identity_fixture
    from training_helpers import ToyLoss,examples,bundle_for
    from asr.checkpoint import save_checkpoint,load_checkpoint,restore_training,TrainingObjects
    from asr.training import create_optimizer,compute_budget,run_window
    from asr.schedule import create_scheduler
    original,state=training_fixture(local_config)
    path=save_checkpoint(local_config.paths.project_root/'runs/r','first',state)
    expected_rng=(random.random(),np.random.random(),torch.rand(3))
    baseline=copy.deepcopy(state.progress)
    run_window(original.bundle,original.optimizer,original.scheduler,None,examples(12),baseline)
    expected=original.bundle.model.lora_A.detach().clone()
    # Reconstruction deliberately consumes every global RNG before restoration.
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    reconstructed=bundle_for(ToyLoss()); opt=create_optimizer(reconstructed,local_config)
    sched=create_scheduler(opt,compute_budget(32))
    objects=TrainingObjects(reconstructed,opt,sched,None)
    progress=restore_training(load_checkpoint(path,identity_fixture()),objects)
    assert random.random()==expected_rng[0]
    assert np.random.random()==expected_rng[1]
    assert torch.equal(torch.rand(3),expected_rng[2])
    run_window(reconstructed,opt,sched,None,examples(12),progress)
    torch.testing.assert_close(reconstructed.model.lora_A,expected,rtol=0,atol=0)
    assert progress.as_dict()==baseline.as_dict()
    assert sched.state_dict()==original.scheduler.state_dict()
