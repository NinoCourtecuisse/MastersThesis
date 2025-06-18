import torch
from torch.distributions import Distribution as D
from torchkde import KernelDensity

class IndependentPrior:
    def __init__(self, dists: list[D]):
        self.dists = dists
        self.dim = len(dists)

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        samples = [dist.sample((n_samples,)) for dist in self.dists]
        return torch.stack(samples, dim=-1).squeeze()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_probs = []
        for i, d in enumerate(self.dists):
            if isinstance(d, D.Uniform):
                mask = (x[:, i] >= d.low) & (x[:, i] < d.high)
            else:
                mask = d.support.check(x[:, i])         # Check if x is in the support
            lp = torch.full_like(x[:, i], -10**9)   # If not: set the log_prob to low value
            lp[mask] = d.log_prob(x[:, i][mask])

            log_probs.append(lp)
        return torch.stack(log_probs, dim=-1).sum(dim=-1)

class ScaledBeta:
    def __init__(self, alpha, beta, low, high):
        self.beta_dist = D.Beta(alpha, beta)
        self.low = low
        self.high = high

    def sample(self, sample_shape):
        u = self.beta_dist.sample(sample_shape)
        return self.low + (self.high - self.low) * u

    def rsample(self, sample_shape):
        u = self.beta_dist.rsample(sample_shape)
        return self.low + (self.high - self.low) * u

    def log_prob(self, x):
        # Map x back to [0,1]
        u = (x - self.low) / (self.high - self.low)
        # Clamp to avoid numerical issues if exactly on boundaries
        u = u.clamp(1e-6, 1.0 - 1e-6)
        return self.beta_dist.log_prob(u) - torch.log(self.high - self.low)

    @property
    def support(self):
        # To match Uniform convention
        return torch.distributions.constraints.interval(self.low, self.high)
