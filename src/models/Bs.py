import torch
from torch import distributions as D

from src.utils.priors import IndependentPrior
from src.models import Model

class Bs(Model):
    def __init__(self, dt:float|torch.Tensor, prior:IndependentPrior):
        super().__init__(dt, prior)

    def log_transition(self, u_params, s, s_next):
        params = self.transform.to(u_params)
        mu, sigma = params.T.unsqueeze(2)
        dt = self.dt

        mean = torch.log(s) + (mu - 0.5 * sigma**2) * dt
        var = sigma**2 * dt
        log_transition = D.Normal(
            loc = mean,
            scale = torch.sqrt(var)
        ).log_prob(torch.log(s_next))
        return log_transition

    def simulate(self, c_params, s0, T, M):
        with torch.no_grad():
            dt = self.dt
            mu, sigma = c_params.T.unsqueeze(2)

            n = int(torch.round(T / dt).item())
            s = torch.zeros((n + 1, M))
            s[0, :] = s0
            Z = D.Normal(loc=0, scale=1).sample(sample_shape=torch.Size((n, M)))
            log_increments = (mu - 0.5 * sigma**2) * dt + sigma * dt**0.5 * Z
            log_price = torch.log(s0) + torch.cumsum(log_increments, dim=0)
            s[1:, :] = torch.exp(log_price)
        return s
