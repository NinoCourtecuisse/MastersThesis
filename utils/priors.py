import torch
from torch import distributions as D
from torchkde import KernelDensity

class IndependentPrior:
    def __init__(self, dists: list[D.Distribution]):
        self.dists = dists
        self.dim = len(dists)

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        samples = [dist.sample((n_samples,)) for dist in self.dists]
        samples = torch.stack(samples, dim=-1)
        if n_samples == 1:
            return samples
        return samples.squeeze()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_probs = []
        for i, d in enumerate(self.dists):
            if isinstance(d, D.Uniform):
                mask = (x[:, i] >= d.low) & (x[:, i] < d.high)
            else:
                mask = d.support.check(x[:, i])         # Check if x is in the support
            lp = torch.full_like(x[:, i], -10**9)       # If not: set the log_prob to low value
            lp[mask] = d.log_prob(x[:, i][mask])

            log_probs.append(lp)
        return torch.stack(log_probs, dim=-1).sum(dim=-1)

class CevPrior:
    def __init__(self, mu_dist, beta_dist, delta_given_beta_fn=None, v=0.2, S=1000):
        self.mu_dist = mu_dist
        self.beta_dist = beta_dist
        self.a = torch.tensor(v * S).log()
        self.b = -torch.tensor(S).log()/2

        if delta_given_beta_fn is None:
            self.delta_given_beta_fn = self._default_delta_given_beta
        else:
            self.delta_given_beta_fn = delta_given_beta_fn

    def _default_delta_given_beta(self, beta):
        loc = self.a + self.b * beta
        return D.LogNormal(loc, 0.5)

    def sample(self, n_samples=1):
        mu = self.mu_dist.sample((n_samples,))
        beta = self.beta_dist.sample((n_samples,))
        delta = self.delta_given_beta_fn(beta).sample()
        samples = torch.stack([mu, delta, beta], dim=-1)
        if n_samples == 1:
            return samples
        return samples.squeeze()

    def log_prob(self, x):
        mu, delta, beta = x[:, 0], x[:, 1], x[:, 2]

        logp_mu = self.mu_dist.log_prob(mu)
        logp_beta = self.beta_dist.log_prob(beta)
        logp_delta_given_beta = self.delta_given_beta_fn(beta).log_prob(delta)

        return logp_mu + logp_delta_given_beta + logp_beta

class KdePrior(KernelDensity):
    def __init__(self, particles, bandwidth):
        super().__init__(kernel='gaussian', bandwidth=bandwidth)
        self.fit(particles)

    def log_prob(self, eval):
        log_prob = self.score_samples(eval)
        return torch.relu(log_prob + 15.0) - 15.0
