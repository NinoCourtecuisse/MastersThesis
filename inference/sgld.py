import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math

class SGLD(Optimizer):
    def __init__(self, params, lr):
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive. Found {lr}.")
        defaults = dict(lr=lr)
        super(SGLD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad_p = p.grad.data
                noise = math.sqrt(lr) * torch.randn_like(p.data)

                p.data.add_(grad_p, alpha=-0.5 * lr)
                p.data.add_(noise)

class SGLDSampler:
    def __init__(self, loss_fn, theta_init:torch.Tensor,
                 burnin:int, lr:float, lr_min:float, gamma:float, n_steps:int):
        """
        SGLD Sampler, starting at theta_init (1, d).
        """
        self.loss_fn = loss_fn
        self.theta = theta_init.detach().clone().requires_grad_(True)
        self.d = self.theta.shape[1]    # Dimension of theta
        self.burnin_period = burnin
        self.ready_to_sample = False    # First need to burnin

        self.sgld = SGLD(params=[self.theta], lr=lr)
        self.scheduler = PowerLRScheduler(self.sgld, gamma, lr_min, n_steps)

    def burnin(self):
        for _ in range(self.burnin_period):
            self.sgld.zero_grad()
            loss = self.loss_fn(self.theta)
            loss.backward()
            self.sgld.step()
            self.scheduler.step()
        self.ready_to_sample = True

    def sample(self, n_samples:int=1):
        if not self.ready_to_sample:
            self.burnin()

        samples = torch.zeros(size=(n_samples, self.d))
        for i in range(n_samples):
            self.sgld.zero_grad()
            loss = self.loss_fn(self.theta)
            loss.backward()
            self.sgld.step()
            self.scheduler.step()

            samples[i, :] = self.theta.detach().squeeze()
        return samples

class PowerInterpolationLRScheduler(_LRScheduler):
    """
    LR Schedule a + (1 - b * t)^\gamma * c where a, b, c are chosen such that
    1. The initial lr, base_lr, is the one provided by the user in the optimizer.
    2. The minimum lr, lr_min, is attained after n_steps steps.
    3. The lr is kept constant below lr_min.
    """
    def __init__(self, optimizer, gamma, lr_min, n_steps, last_epoch=-1):
        self.gamma = gamma
        self.lr_min = lr_min
        self.n_steps = n_steps

        base_lrs = [group['lr'] for group in optimizer.param_groups]
        assert all(lr == base_lrs[0] for lr in base_lrs), \
            "All param groups must have the same base learning rate"
        self.base_lr = base_lrs[0]
        self.lr_diff = self.base_lr - self.lr_min

        self.a = self.lr_min
        self.b = 1 / self.n_steps
        self.c = self.lr_diff
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        if t < self.n_steps:
            lr = self.a + (1 - self.b * t)**(self.gamma) * self.c
        else:
            lr = self.lr_min
        return [lr for _ in self.optimizer.param_groups]

class PowerLRScheduler(_LRScheduler):
    """
    LR Schedule a(b + t)^{-\gamma} where a, b are chosen such that
    1. The initial lr, base_lr, is the one provided by the user in the optimizer.
    2. The minimum lr, lr_min, is attained after n_steps steps.
    3. The lr is kept constant below lr_min.
    """
    def __init__(self, optimizer, gamma, lr_min, n_steps, last_epoch=-1):
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
