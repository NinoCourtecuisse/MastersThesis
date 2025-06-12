import torch
from torch.distributions import Distribution
from torchkde import KernelDensity

class IndependentPrior:
    def __init__(self, dists: list[Distribution]):
        self.dists = dists
        self.dim = len(dists)

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        samples = [dist.sample((n_samples,)) for dist in self.dists]
        return torch.stack(samples, dim=-1).squeeze()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_probs = []
        for i, d in enumerate(self.dists):
            mask = d.support.check(x[:, i])         # Check if x is in the support
            lp = torch.full_like(x[:, i], -10**6)   # If not: set the log_prob to low value
            lp[mask] = d.log_prob(x[:, i][mask])

            log_probs.append(lp)
        return torch.stack(log_probs, dim=-1).sum(dim=-1)
