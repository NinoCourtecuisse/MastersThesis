import torch
from torch import distributions as D
from src.utils.priors import IndependentPrior, CevPrior, NigPrior
from src.utils.special_functions import logit, inv_logit
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
    
    def log_abs_det(self, unconstrained_x):
        log_abs_dets = torch.zeros(size=(unconstrained_x.shape[0],))
        for i, transform in enumerate(self.transforms):
            grad_i = torch.vmap(torch.func.grad(transform, argnums=0))(unconstrained_x[:, i])
            log_abs_dets += grad_i.abs().log()
        return log_abs_dets

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
        self.eta_transform = D.transform_to(prior.eta_dist.support)
        self.xi_transform = T.identity_transform

    def to(self, unconstrained_x: torch.Tensor) -> torch.Tensor:
        u_mu, u_sigma, u_xi, u_eta = unconstrained_x.T
        c_mu = self.mu_transform(u_mu)
        c_sigma = self.sigma_transform(u_sigma)
        c_xi = self.xi_transform(u_xi)
        c_eta = self.eta_transform(u_eta)
        return torch.stack([c_mu, c_sigma, c_xi, c_eta], dim=-1)

    def inv(self, constrained_x: torch.Tensor) -> torch.Tensor:
        c_mu, c_sigma, c_xi, c_eta = constrained_x.T
        u_mu = self.mu_transform.inv(c_mu)
        u_sigma = self.sigma_transform.inv(c_sigma)
        u_xi = self.xi_transform.inv(c_xi)
        u_eta = self.eta_transform.inv(c_eta)
        return torch.stack([u_mu, u_sigma, u_xi, u_eta], dim=-1)

class SvTmbTransform:
    def to(self, constrained_x):
        mu, sigma_y, sigma_h, phi, rho = constrained_x.T
        log_sigma_y = torch.log(sigma_y)
        log_sigma_h = torch.log(sigma_h)
        logit_phi = logit(phi)
        logit_rho = logit(rho)

        tmb_params = torch.stack([mu, log_sigma_y, log_sigma_h, logit_phi, logit_rho], dim=1)
        return tmb_params

    def inv(self, tmb_x):
        mu, log_sigma_y, log_sigma_h, logit_phi, logit_rho = tmb_x.T
        c_params = torch.stack([
            mu, log_sigma_y.exp(), log_sigma_h.exp(),
            inv_logit(logit_phi), inv_logit(logit_rho)
        ], dim=1)
        return c_params

    def log_abs_det(self, constrained_x):
        mu, sigma_y, sigma_h, phi, rho = constrained_x.T
        grad_mu = torch.ones_like(mu)
        grad_sigma_y = 1 / sigma_y
        grad_sigma_h = 1 / sigma_h
        grad_phi = 2 / ((1 + phi) * (1 - phi))
        grad_rho = 2 / ((1 + rho) * (1 - rho))
        grads = [grad_mu, grad_sigma_y, grad_sigma_h, grad_phi, grad_rho]
        log_abs_dets = torch.zeros(size=(constrained_x.shape[0],))
        for i in range(len(grads)):
            log_abs_dets += grads[i].abs().log()
        return log_abs_dets

class SabrTmbTransform:
    def to(self, constrained_x):
        mu, beta, sigma, rho = constrained_x.T
        log_beta = torch.log(beta)
        log_sigma = torch.log(sigma)
        logit_rho = logit(rho)

        tmb_params = torch.stack([mu, log_beta, log_sigma, logit_rho], dim=1)
        return tmb_params
    
    def inv(self, tmb_x):
        mu, log_beta, log_sigma, logit_rho = tmb_x.T
        c_params = torch.stack([
            mu, log_beta.exp(), log_sigma.exp(), inv_logit(logit_rho)
        ], dim=1)
        return c_params

    def log_abs_det(self, constrained_x):
        mu, beta, sigma, rho = constrained_x.T
        grad_mu = torch.ones_like(mu)
        grad_beta = 1 / beta
        grad_sigma = 1 / sigma
        grad_rho = 2 / ((1 + rho) * (1 - rho))
        grads = [grad_mu, grad_beta, grad_sigma, grad_rho]
        log_abs_dets = torch.zeros(size=(constrained_x.shape[0],))
        for i in range(len(grads)):
            log_abs_dets += grads[i].abs().log()
        return log_abs_dets
