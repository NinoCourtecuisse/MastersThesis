import math
import torch
from torch.utils.data import DataLoader

from src.inference import SGLD
from src.inference.lrSchedule import PowerLRScheduler
from src.models import Model

import itertools
def get_batch_at_index(dataloader, i):
    if i < 0 or i >= len(dataloader):
        raise IndexError(f"Batch index {i} out of range (0 ≤ i < {len(dataloader)})")
    return next(itertools.islice(dataloader, i, i+1))

class ModelPool:
    """
    Class representing a pool of models: several model classes with several parameters.

    Args:
        model_classes (list[Model]): List of M model classes, e.g. Bs, Cev, Nig.
        init_prior (torch.Tensor): Tensor of shape (M,), representing the prior on the model classes.
        init_particles list[torch.Tensor]): List of M tensors of shape (N, D_m) where N is the number
            of particles in each model class, and D_M is the dimension of a parameter vector of model class m.
    """
    def __init__(self, model_classes:list[Model], init_log_prior:torch.Tensor,
                 init_particles:list[torch.Tensor]):
        self.model_classes = model_classes
        self.log_prior = init_log_prior
        self.init_log_prior = init_log_prior
        self.particles = init_particles

        self.M = len(self.model_classes)
        self.N = len(self.particles[0])

    def estimate_nothing(self, data:torch.Tensor, temperature:float=1.0) -> torch.Tensor:
        """
        Recursively compute the (log) weights given by "Estimate Nothing"
        based on the data (S_0, ..., S_T):
            log π_t(θ^(i,m), M_m)
        for all models m=1, ..., M, i=1, ..., N and all times t=1, ..., T

        Args:
            data (torch.Tensor): Tensor of shape (T+1,) of asset prices.

        Returns:
            torch.Tensor: Tensor of shape (M, T, N) where each entry (m, t, i)
            represents log π_t(θ^(i,m), M_m).
            torch.Tensor: Tensor of shape (M, N) of log importance weights at time
            t = T (only useful for resampling step).
        """
        T = len(data) - 1

        # Evaluate the log likelihood recursively
        ll = torch.zeros(size=(self.M, T+1, self.N))
        for t in range(1, T+1):
            for m in range(self.M):
                model_class = self.model_classes[m]
                theta = self.particles[m]
                ll[m, t, :] = ll[m, t-1, :] + model_class.log_transition(theta, data[t-1], data[t]).squeeze()

        ll = ll[:, 1:, :]   # Get rid of t=0, it was just to instantiate the recursion.
        ll = temperature * ll
        # Compute the log evidences
        log_evidence = torch.logsumexp(ll, dim=2, keepdim=True) - math.log(self.N)     # (M, T, 1)

        # Compute the log importance weights
        log_iw = ll - (log_evidence + math.log(self.N))     # (M, T, N)

        # Compute the model classes log posterior
        unnormalized_log_post = log_evidence + self.log_prior.view(self.M, 1, 1)    # (M, T, 1)
        log_post = unnormalized_log_post - torch.logsumexp(unnormalized_log_post, dim=0, keepdim=True)    # (M, T, 1)

        # Compute "Estimate Nothing" log weights
        log_pi = log_iw + log_post      # (M, T, N)

        final_log_iw = log_iw[:, -1, :]
        return log_pi, final_log_iw, log_post.squeeze(), ll + self.log_prior.view(self.M, 1, 1)

    def update_particles(self, dataloader:DataLoader, log_iw:torch.Tensor,
                         n_grad_steps:int, lrs:list[float], batch_weights:torch.Tensor):
        """
        Update all the particles in two steps:
        1. Resample based on the importance weights.
        2. Move towards the MAP with SGLD.

        Args:
            log_iw (torch.Tensor): Tensor of shape (M, N) of log importance weights.
            lrs (list[float]): List of learning rate for each model class.
        Returns:
        """
        new_particles = [None] * self.M
        n_batch = len(dataloader)

        for m in range(self.M):
            model_class = self.model_classes[m]
            particles = self.particles[m]   # (N, D_m)

            # 1. Resample
            iw = log_iw[m, :].exp()
            indices = torch.multinomial(iw, num_samples=self.N, replacement=True)
            particles = particles[indices, :]

            # 2. Move
            params = particles.detach().clone().requires_grad_(True)
            optimizer = SGLD(params=[params], lr=lrs[m])
            scheduler = PowerLRScheduler(optimizer, gamma=0.51,
                                         lr_min=lrs[m]/10, n_steps=10**3)
            for n in range(n_grad_steps):
                optimizer.zero_grad()

                # Choose batch of data
                batch_idx = torch.multinomial(batch_weights, num_samples=1)
                data = get_batch_at_index(dataloader, batch_idx)[0]

                batch_ll = torch.sum(
                    model_class.log_transition(params, data[:-1], data[1:]),
                    dim=1
                )
                batch_lprior = model_class.prior.log_prob(model_class.transform.to(params))
                loss = -(batch_lprior + n_batch * batch_ll).sum()
                loss.backward()

                optimizer.step()
                scheduler.step()

            particles = params.detach().clone()
            new_particles[m] = particles

        self.particles = new_particles
        return
