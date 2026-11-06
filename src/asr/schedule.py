"""LR is indexed by completed successful optimizer calls, including LR-zero calls."""
from torch.optim.lr_scheduler import LambdaLR


def create_scheduler(optimizer,budget):
    warmup,total = budget.warmup_updates,budget.total_updates
    def multiplier(update):
        if update < warmup:
            return update/max(1,warmup)
        return max(0.0,(total-update)/max(1,total-warmup))
    return LambdaLR(optimizer,multiplier,last_epoch=-1)
