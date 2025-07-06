import torch
from torch import distributions as D
from utils.priors import IndependentPrior, CevPrior, NigPrior
import torch.distributions.transforms as T

class IndependentTransform:
    def __init__(self, prior: IndependentPrior):
        self.transforms = []
        for i, d in enumerate(prior.dists):
            constraint = d.support
            transform = D.transform_to(constraint)
            self.transforms.append(transform)
    
    def to(self, unconstrained_x: torch.Tensor) -> torch.Tensor:
        constrained_x = []
        for i, transform in enumerate(self.transforms):
            constrained_x.append(transform(unconstrained_x[:, i]))
        return torch.stack(constrained_x, dim=-1)
    
    def inv(self, constrained_x: torch.Tensor) -> torch.Tensor:
        unconstrained_x = []
        for i, transform in enumerate(self.transforms):
            unconstrained_x.append(transform.inv(constrained_x[:, i]))
        return torch.stack(unconstrained_x, dim=-1)

class CevTransform:
    def __init__(self, prior: CevPrior):
        self.mu_transform = D.transform_to(prior.mu_dist.support)
        self.beta_transform = D.transform_to(prior.beta_dist.support)
        self.delta_transform = T.ExpTransform()

        self.a = prior.a
        self.b = prior.b

    def to(self, unconstrained_x: torch.Tensor) -> torch.Tensor:
        u_mu, u_delta, u_beta = unconstrained_x[:, 0], unconstrained_x[:, 1], unconstrained_x[:, 2]
        c_mu = self.mu_transform(u_mu)
        c_beta = self.beta_transform(u_beta)
        c_delta = self.delta_transform(u_delta + self.a + self.b * c_beta)
        return torch.stack([c_mu, c_delta, c_beta], dim=-1)

    def inv(self, constrained_x: torch.Tensor) -> torch.Tensor:
        c_mu, c_delta, c_beta = constrained_x[:, 0], constrained_x[:, 1], constrained_x[:, 2]
        u_mu = self.mu_transform.inv(c_mu)
        u_beta = self.beta_transform.inv(c_beta)
        u_delta = self.delta_transform.inv(c_delta) - (self.a + self.b * c_beta)
        return torch.stack([u_mu, u_delta, u_beta], dim=-1)

class NigTransform:
    def __init__(self, prior: NigPrior):
        self.mu_transform = D.transform_to(prior.mu_dist.support)
        self.sigma_transform = D.transform_to(prior.sigma_dist.support)
        self.gamma1_transform = D.transform_to(prior.gamma1_dist.support)
        self.eps = 1e-10  # to enforce gamma2 > gamma1 strictly
        self.gamma2_high = prior.gamma2_high

    def _gamma2_transform(self, gamma1: torch.Tensor) -> T.ComposeTransform:
        low = 5 * gamma1**2 / 3 + self.eps
        high = self.gamma2_high
        return T.ComposeTransform([
            T.SigmoidTransform(),      # R -> (0, 1)
            T.AffineTransform(loc=low, scale=high - low, event_dim=0)   # (0, 1) -> (low, high)
        ])

    def to(self, unconstrained_x: torch.Tensor) -> torch.Tensor:
        u_mu, u_sigma, u_gamma1, u_gamma2 = unconstrained_x.T

        mu = self.mu_transform(u_mu)
        sigma = self.sigma_transform(u_sigma)
        gamma1 = self.gamma1_transform(u_gamma1)
        gamma2_transform = self._gamma2_transform(gamma1)
        gamma2 = gamma2_transform(u_gamma2)
        return torch.stack([mu, sigma, gamma1, gamma2], dim=-1)

    def inv(self, constrained_x: torch.Tensor) -> torch.Tensor:
        mu, sigma, gamma1, gamma2 = constrained_x.T

        u_mu = self.mu_transform.inv(mu)
        u_sigma = self.sigma_transform.inv(sigma)
        u_gamma1 = self.gamma1_transform.inv(gamma1)
        gamma2_transform = self._gamma2_transform(gamma1)
        u_gamma2 = gamma2_transform.inv(gamma2)  
        return torch.stack([u_mu, u_sigma, u_gamma1, u_gamma2], dim=-1)
