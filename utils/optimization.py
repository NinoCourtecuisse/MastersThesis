import torch
from torch import distributions as D
from utils.priors import IndependentPrior, CevPrior
from torch.distributions.transforms import ExpTransform

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
        self.delta_transform = ExpTransform()  # Because LogNormal support is positive

        self.a = prior.a
        self.b = prior.b

    def to(self, unconstrained_x: torch.Tensor) -> torch.Tensor:
        """
        Maps unconstrained_x (in R^3) to constrained values (mu, delta, beta)
        """
        u_mu, u_delta, u_beta = unconstrained_x[:, 0], unconstrained_x[:, 1], unconstrained_x[:, 2]

        c_mu = self.mu_transform(u_mu)
        c_beta = self.beta_transform(u_beta)

        # LogNormal: delta = exp(u_delta) * exp(loc), i.e., exp(u_delta + a + b * beta)
        c_delta = self.delta_transform(u_delta + self.a + self.b * c_beta)

        return torch.stack([c_mu, c_delta, c_beta], dim=-1)

    def inv(self, constrained_x: torch.Tensor) -> torch.Tensor:
        """
        Maps constrained values (mu, delta, beta) to unconstrained_x in R^3
        """
        c_mu, c_delta, c_beta = constrained_x[:, 0], constrained_x[:, 1], constrained_x[:, 2]

        u_mu = self.mu_transform.inv(c_mu)
        u_beta = self.beta_transform.inv(c_beta)

        # Invert: u_delta = log(delta) - (a + b * beta)
        u_delta = self.delta_transform.inv(c_delta) - (self.a + self.b * c_beta)

        return torch.stack([u_mu, u_delta, u_beta], dim=-1)
