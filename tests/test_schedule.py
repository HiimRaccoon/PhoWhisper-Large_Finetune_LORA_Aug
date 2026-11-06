import copy
import pytest
import torch


def test_budget_and_lr_boundaries_are_exact():
    from asr.training import compute_budget
    from asr.schedule import create_scheduler
    budget=compute_budget(5036)
    assert (budget.updates_per_epoch,budget.total_updates,budget.warmup_updates)==(315,3150,158)
    parameter=torch.nn.Parameter(torch.ones(1))
    optimizer=torch.optim.AdamW([parameter],lr=1e-4)
    scheduler=create_scheduler(optimizer,budget)
    expected={0:0,1:1e-4/158,157:1e-4*157/158,158:1e-4,159:1e-4*2991/2992,3149:1e-4/2992}
    for u in range(3150):
        if u in expected: assert optimizer.param_groups[0]['lr']==pytest.approx(expected[u],abs=1e-15)
        parameter.grad=torch.ones_like(parameter)
        optimizer.step(); scheduler.step()
    assert scheduler.last_epoch==3150
    assert optimizer.param_groups[0]['lr']==0


@pytest.mark.parametrize('cut',[157,158,159,3149])
def test_scheduler_resume_preserves_the_remaining_lr_trace(cut):
    from asr.training import compute_budget
    from asr.schedule import create_scheduler
    budget=compute_budget(5036)
    p=torch.nn.Parameter(torch.ones(1)); opt=torch.optim.AdamW([p],lr=1e-4); sched=create_scheduler(opt,budget)
    for _ in range(cut): p.grad=torch.ones(1); opt.step(); sched.step()
    state=copy.deepcopy((opt.state_dict(),sched.state_dict()))
    expected=[]
    for _ in range(cut,3150):
        expected.append(opt.param_groups[0]['lr']); p.grad=torch.ones(1); opt.step(); sched.step()
    q=torch.nn.Parameter(torch.ones(1)); resumed=torch.optim.AdamW([q],lr=1e-4)
    rs=create_scheduler(resumed,budget); resumed.load_state_dict(state[0]); rs.load_state_dict(state[1])
    observed=[]
    for _ in range(cut,3150):
        observed.append(resumed.param_groups[0]['lr']); q.grad=torch.ones(1); resumed.step(); rs.step()
    assert observed==expected
