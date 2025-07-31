from abc import ABC, abstractmethod

import torch
from torch import distributions as D

from src.utils.optimization import IndependentTransform, CevTransform, \
                                    NigTransform

"""
Defines structured priors with possible dependencies.

- IndependentPrior: Independent priors on each dimension: p(x₁, ..., xₙ) = ∏ p(xᵢ)
- CevPrior: Custom dependency for CEV model: p(μ, δ, β) = p(μ) p(δ | β) p(β)
- NigPrior: Custom dependency for NIG model: p(μ, σ, ξ, η) = p(μ) p(σ) p(ξ | η) p(η)
"""

class Prior(ABC):
    def __init__(self, transform):
        self.transform = transform

    @abstractmethod
    def sample(self, n_samples:int=1) -> torch.Tensor:
        pass

    @abstractmethod
    def log_prob(self, x:torch.Tensor) -> torch.Tensor:
        pass

class IndependentPrior(Prior):
    """
    Defines a prior of the form: p(x₁, ..., xₙ) = ∏ p(xᵢ)

    Args:
        dists (list[D.Distribution]): List of distributions, i.e. p(xᵢ) for i=1, ..., n.
    """
    def __init__(self, dists: list[D.Distribution]):
        self.dists = dists
        self.dim = len(dists)

        transform = IndependentTransform(self)
        super().__init__(transform)

    def sample(self, n_samples=1):
        samples = [dist.sample((n_samples,)) for dist in self.dists]
        samples = torch.stack(samples, dim=-1)
        if n_samples == 1:
            return samples
        return samples.squeeze()

    def log_prob(self, x):
        log_probs = []
        for i, d in enumerate(self.dists):
            lp = d.log_prob(x[:, i])
            log_probs.append(lp)
        return torch.stack(log_probs, dim=-1).sum(dim=-1)

class CevPrior(Prior):
    """
    Defines a prior of the form: p(μ, δ, β) = p(μ) p(δ | β) p(β)
    where p(δ | β) = N(a + b * β, 1.0).
    The parameters a and b are derived from the local volatility structure:
        a = log(v * S)
        b = -log(S) / 2
    where S is a typical value of the asset price and v is a typical volatility level.

    Args:
        mu_dist (D.Distribution): Prior distribution on mu, i.e. p(μ).
        beta_dist (D.Distribution): Prior distribution on beta, i.e. p(β).
        v (float): Typical value for the volatility (default: 0.2 for SP500).
        S (float): Typical value for the asset price (default: 1000 for SP500).
    """

    def __init__(self, mu_dist:D.Distribution, beta_dist:D.Distribution,
                 v:float=0.2, S:float=1000):
        self.mu_dist = mu_dist
        self.beta_dist = beta_dist
        self.a = torch.tensor(v * S).log()
        self.b = -torch.tensor(S).log()/2

        transform = CevTransform(self)
        super().__init__(transform)

    def _default_delta_given_beta(self, beta):
        loc = self.a + self.b * beta
        return D.LogNormal(loc, 1.0)

    def sample(self, n_samples=1):
        mu = self.mu_dist.sample((n_samples,))
        beta = self.beta_dist.sample((n_samples,))
        delta = self._default_delta_given_beta(beta).sample()
        samples = torch.stack([mu, delta, beta], dim=-1)
        if n_samples == 1:
            return samples
        return samples.squeeze()

    def log_prob(self, x):
        mu, delta, beta = x.T

        logp_mu = self.mu_dist.log_prob(mu)
        logp_beta = self.beta_dist.log_prob(beta)
        logp_delta_given_beta = self._default_delta_given_beta(beta).log_prob(delta)
        return logp_mu + logp_delta_given_beta + logp_beta

class NigPrior(Prior):
    """
    Defines a prior of the form: p(μ, σ, ξ, η) = p(μ) p(σ) p(ξ | η) p(η)
    Based on https://arxiv.org/abs/2203.05510, we choose
        - p(η) = Exp(θ_η)
        - p(ξ | η) = Laplace(xi_loc, θ_ξ * √η)

    Args:
        mu_dist (D.Distribution): Prior distribution on mu, i.e. p(μ).
        sigma_dist (D.Distribution): Prior distribution on sigma, i.e. p(σ).
        theta_eta (float, positive): Rate for exponential prior on η.
        theta_xi (float, positive): Factor in the rate for Laplace prior on ξ.
    """

    def __init__(self, mu_dist:D.Distribution, sigma_dist:D.Distribution,
                 theta_eta:float, theta_xi:float):
        self.mu_dist = mu_dist
        self.sigma_dist = sigma_dist
        self.eta_dist = D.Exponential(rate=theta_eta)

        self.theta_xi = theta_xi
        self.xi_rate_min = self.theta_xi * torch.tensor(1e-1).sqrt()
        self.xi_loc = 0.0

        transform = NigTransform(self)
        super().__init__(transform)

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
