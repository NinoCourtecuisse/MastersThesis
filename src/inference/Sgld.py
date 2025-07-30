import torch
from torch.optim import Optimizer
import math

from src.inference.lrSchedule import PowerLRScheduler

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
