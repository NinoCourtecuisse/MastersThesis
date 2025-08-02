from torch.optim.lr_scheduler import _LRScheduler

class PowerLRScheduler(_LRScheduler):
    """
    LR Schedule of the form
        a(b + t)^{-\gamma}
    used with SGLD.

    a, b are chosen such that
    1. The initial lr, is the one provided by the user in the optimizer.
    2. The minimum lr, is attained after n_steps steps.
    3. The lr is kept constant below lr_min.
    """
    def __init__(self, optimizer, gamma:float, lr_min:float, n_steps:int, last_epoch=-1):
        self.gamma = gamma
        self.lr_min = lr_min
        self.n_steps = n_steps

        base_lrs = [group['lr'] for group in optimizer.param_groups]
        assert all(lr == base_lrs[0] for lr in base_lrs), \
            "All param groups must have the same base learning rate"
        self.base_lr = base_lrs[0]
        self.lr_ratio = self.lr_min / self.base_lr

        self.b = self.n_steps * self.lr_ratio**(1/self.gamma) / (1 - self.lr_ratio**(1/self.gamma))
        self.a = self.base_lr * self.b**self.gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch + 1

        if t < self.n_steps:
            lr = self.a * (self.b + t)**(-self.gamma)
        else:
            lr = self.lr_min
        return [lr for _ in self.optimizer.param_groups]
