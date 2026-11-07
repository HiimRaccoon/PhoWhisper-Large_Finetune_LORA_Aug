import copy

import pytest
import torch


def test_restore_rejects_reordered_same_shape_optimizer_parameters(local_config):
    from training_helpers import ToyLoss, bundle_for, examples
    from test_checkpoint import identity_fixture
    from asr.checkpoint import TrainingObjects, capture_training, restore_training
    from asr.config import canonical_hash
    from asr.contracts import TrainingProgress
    from asr.training import create_optimizer, compute_budget, run_window
    from asr.schedule import create_scheduler

    class TwoParameters(ToyLoss):
        def __init__(self):
            super().__init__()
            self.lora_B = torch.nn.Parameter(torch.tensor([[0.3]]))

        def forward(self, **kwargs):
            output = super().forward(**kwargs)
            output.loss = output.loss + self.lora_B.square().sum()
            return output

    def factory():
        bundle = bundle_for(TwoParameters())
        optimizer = create_optimizer(bundle, local_config)
        return TrainingObjects(bundle, optimizer, create_scheduler(optimizer, compute_budget(32)), None)

    original = factory()
    order = list(range(32))
    progress = TrainingProgress(permutation_hash=canonical_hash(order))
    run_window(original.bundle, original.optimizer, original.scheduler, None, examples(16), progress)
    state = capture_training(original, progress, identity_fixture(), order)
    resumed = factory()
    resumed.optimizer.param_groups[0]['params'].reverse()
    before = copy.deepcopy(resumed.bundle.model.state_dict())
    with pytest.raises(ValueError, match='optimizer parameter'):
        restore_training(state, resumed)
    for name, tensor in resumed.bundle.model.state_dict().items():
        assert torch.equal(tensor, before[name])
