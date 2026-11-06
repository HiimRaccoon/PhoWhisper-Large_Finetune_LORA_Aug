import pytest
import torch
from training_helpers import ToyLoss,examples,bundle_for


def setup(local_config):
    from asr.training import compute_budget,create_optimizer
    from asr.schedule import create_scheduler
    from asr.contracts import TrainingProgress
    bundle=bundle_for(ToyLoss()); opt=create_optimizer(bundle,local_config)
    return bundle,opt,create_scheduler(opt,compute_budget(5036)),TrainingProgress()


def test_zero_lr_success_advances_moments_and_progress(local_config):
    from asr.training import run_window
    bundle,opt,sched,progress=setup(local_config)
    before=bundle.model.lora_A.detach().clone()
    result=run_window(bundle,opt,sched,None,examples(1),progress)
    assert torch.equal(before,bundle.model.lora_A)
    assert opt.state[bundle.model.lora_A]['step'].item()==1
    assert opt.state[bundle.model.lora_A]['exp_avg'].abs().sum()>0
    assert progress.global_update==1 and result.optimizer_step_invoked


def test_nonfinite_loss_does_not_advance_optimizer_or_scheduler(local_config):
    from asr.training import run_window
    bundle,opt,sched,progress=setup(local_config)
    batch=examples(1); batch[0].input_features.fill_(float('inf'))
    with pytest.raises(FloatingPointError): run_window(bundle,opt,sched,None,batch,progress)
    assert progress.global_update==0 and sched.last_epoch==0 and not opt.state


def test_scaler_skip_cannot_be_mistaken_for_none_return_success(local_config):
    from asr.training import run_window
    class SkipScaler:
        def scale(self,loss): return loss
        def unscale_(self,opt): pass
        def step(self,opt): return None
        def update(self): pass
    bundle,opt,sched,progress=setup(local_config)
    with pytest.raises(RuntimeError,match='skip'):
        run_window(bundle,opt,sched,SkipScaler(),examples(1),progress)
    assert progress.global_update==0 and sched.last_epoch==0


def test_real_cpu_grad_scaler_none_return_is_still_success(local_config):
    from asr.training import run_window
    bundle,opt,sched,progress=setup(local_config)
    scaler=torch.amp.GradScaler('cpu',init_scale=128)
    result=run_window(bundle,opt,sched,scaler,examples(12),progress)
    assert result.optimizer_step_invoked and progress.global_update==1
    assert progress.global_micro_step==12 and scaler.get_scale()==128


def test_oom_is_reported_without_advancing_or_changing_method(local_config):
    from asr.training import run_window
    class OOM(ToyLoss):
        def forward(self,**kwargs): raise torch.OutOfMemoryError('fixture OOM')
    bundle,opt,sched,progress=setup(local_config)
    bundle.model=OOM()
    with pytest.raises(torch.OutOfMemoryError):
        run_window(bundle,opt,sched,None,examples(1),progress)
    assert progress.global_update==0 and sched.last_epoch==0
    assert local_config.training.micro_batch_size==1
