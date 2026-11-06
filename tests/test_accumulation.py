import copy
import pytest
import torch
from training_helpers import ToyLoss,examples,bundle_for


@pytest.mark.parametrize('k',[16,12])
def test_window_update_equals_mean_of_individual_utterance_losses(local_config,k):
    from asr.training import compute_budget,create_optimizer,run_window
    from asr.schedule import create_scheduler
    from asr.contracts import TrainingProgress
    model=ToyLoss(); reference=copy.deepcopy(model)
    bundle=bundle_for(model); refbundle=bundle_for(reference)
    opt=create_optimizer(bundle,local_config); refopt=create_optimizer(refbundle,local_config)
    sched=create_scheduler(opt,compute_budget(5036)); refsched=create_scheduler(refopt,compute_budget(5036))
    progress=TrainingProgress()
    batch=examples(k)
    for _ in range(2):
        result=run_window(bundle,opt,sched,None,batch,progress)
        refopt.zero_grad(set_to_none=True)
        torch.stack([reference(**e.model_inputs()).loss for e in batch]).mean().backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.0)
        refopt.step(); refsched.step()
        torch.testing.assert_close(model.lora_A,reference.lora_A,rtol=0,atol=0)
        torch.testing.assert_close(opt.state[model.lora_A]['exp_avg'],refopt.state[reference.lora_A]['exp_avg'])
    assert progress.global_update==2 and progress.global_micro_step==2*k
    assert progress.pending_micro_steps==0
    assert result.window_size==k


def test_epoch_shuffle_is_private_and_covers_each_occurrence():
    from asr.training import epoch_order
    torch.manual_seed(10); before=torch.get_rng_state().clone()
    order=epoch_order(5036,0)
    assert sorted(order)==list(range(5036))
    assert order==epoch_order(5036,0) and order!=epoch_order(5036,1)
    assert torch.equal(torch.get_rng_state(),before)
