import torch
from torch import distributions as D

"""
Defines structured priors with possible dependencies.

- IndependentPrior: Independent priors on each dimension: p(x₁, ..., xₙ) = ∏ p(xᵢ)
- CevPrior: Custom dependency for CEV model: p(μ, δ, β) = p(μ) p(δ | β) p(β)
- NigPrior: Custom dependency for NIG model: p(μ, σ, ξ, η) = p(μ) p(σ) p(ξ | η) p(η)
"""

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
            lp = d.log_prob(x[:, i])
            log_probs.append(lp)
        return torch.stack(log_probs, dim=-1).sum(dim=-1)

class CevPrior:
    def __init__(self, mu_dist, beta_dist, v=0.2, S=1000):
        self.mu_dist = mu_dist
        self.beta_dist = beta_dist
        self.a = torch.tensor(v * S).log()
        self.b = -torch.tensor(S).log()/2
        self.delta_given_beta_fn = self._default_delta_given_beta

    def _default_delta_given_beta(self, beta):
        loc = self.a + self.b * beta
        return D.LogNormal(loc, 1.0)

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

class NigPrior:
    def __init__(self, mu_dist, sigma_dist, theta_eta, theta_xi):
        self.mu_dist = mu_dist
        self.sigma_dist = sigma_dist
        self.eta_dist = D.Exponential(rate=theta_eta)

        self.theta_xi = theta_xi
        self.xi_rate_min = self.theta_xi * torch.tensor(1e-1).sqrt()
        self.xi_loc = 0.0

    def xi_given_eta(self, eta):
        rate = self.theta_xi * torch.sqrt(eta)
        cut_rate = torch.clamp(rate, min=self.xi_rate_min)
        return D.Laplace(loc=self.xi_loc, scale=1/cut_rate)

    def sample(self, n_samples=1):
        mu = self.mu_dist.sample((n_samples,))
        sigma = self.sigma_dist.sample((n_samples,))
        eta = self.eta_dist.sample((n_samples,))
        xi = self.xi_given_eta(eta).sample()
        samples = torch.stack([mu, sigma, xi, eta], dim=-1)
        if n_samples == 1:
            return samples
        return samples.squeeze()

    def log_prob(self, x):
        mu, sigma, xi, eta = x.T

        logp_mu = self.mu_dist.log_prob(mu)
        logp_sigma = self.sigma_dist.log_prob(sigma)
        logp_xi = self.xi_given_eta(eta).log_prob(xi)
        logp_eta = self.eta_dist.log_prob(eta)
        return logp_mu + logp_sigma + logp_xi + logp_eta
