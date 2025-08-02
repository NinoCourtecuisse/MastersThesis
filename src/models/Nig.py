import torch

from src.utils.distributions import InverseGaussian, NormalInverseGaussian
from src.utils.priors import NigPrior, IndependentPrior
from src.models import Model

class Nig(Model):
    def __init__(self, dt:float|torch.Tensor, prior:IndependentPrior|NigPrior):
        super().__init__(dt, prior)

    def log_transition(self, u_params, s, s_next):
        params = self.transform.to(u_params)
        mu, sigma, xi, eta = params.T.unsqueeze(2)
        dt = self.dt

        log_return = torch.log(s_next / s)
        log_transition = NormalInverseGaussian.from_moments(
            mu=dt * mu,
            sigma=torch.sqrt(dt) * sigma,
            xi=torch.sqrt(dt) * xi,
            eta=eta / dt
        ).log_prob(log_return)
        return log_transition

    def simulate(self, c_params, s0, T, M):
        with torch.no_grad():
            mu, sigma, xi, eta = c_params.T.unsqueeze(2)
            mu_, alpha, beta, delta = NormalInverseGaussian.reparametrize(mu, sigma, xi, eta)
            dt = self.dt
            n = int(torch.round(T / dt).item())

            # Generate M paths from IG process
            dI = InverseGaussian(
                mu = delta * dt / torch.sqrt(alpha**2 - beta**2),
                lam = (delta * dt)**2
            ).sample(sample_shape=(n - 1, M))
            I = torch.cumsum(dI, dim=0)

            # Generate M paths from NIG process
            Z = torch.randn(size=(n - 1, M))
            W_It = torch.cumsum(torch.sqrt(dI) * Z, dim=0)
            X = mu_ * torch.linspace(dt, T, n - 1).unsqueeze(1) + beta * I + W_It

            s = s0 * torch.ones((n, M))
            s[1:, :] = s0 * torch.exp(X)
        return s
