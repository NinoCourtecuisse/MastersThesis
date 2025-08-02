import torch
from torch.optim import Optimizer
import math

class SGLD(Optimizer):
    """
    Implement the Stochastic Gradient Langevin Dynamics optimizer.
    """
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
