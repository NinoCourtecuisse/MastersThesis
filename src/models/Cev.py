import torch
from torch import distributions as D

from src.utils.priors import CevPrior
from src.models import Model

class Cev(Model):
    def __init__(self, dt:float|torch.Tensor, prior:CevPrior):
        super().__init__(dt, prior)

    def log_transition(self, u_params, s, s_next):
        params = self.transform.to(u_params)
        mu, delta, beta = params.T.unsqueeze(2)
        dt = self.dt

        C = torch.tensor(10**3)
        tmp1 = C**2 * torch.exp((beta - 2) * mu * dt) * torch.ones(size=(len(mu), len(s)))
        tmp2 = s[None, :]**(beta - 2)
        local_var = delta**2 * torch.minimum(tmp1, tmp2)

        mean = torch.log(s) + (mu - 0.5 * local_var) * dt
        var = local_var * dt
        log_transition = D.Normal(
            loc=mean,
            scale=torch.sqrt(var)
        ).log_prob(torch.log(s_next))
        return log_transition

    def simulate(self, c_params, s0, T, M):
        with torch.no_grad():
            dt = self.dt
            mu = c_params[:, 0].unsqueeze(1)
            delta = c_params[:, 1].unsqueeze(1)
            beta = c_params[:, 2].unsqueeze(1)

            n = int(torch.round(T / dt).item())
            s = torch.zeros((n + 1, M))
            s[0, :] = s0
            Z = D.Normal(loc=0, scale=1).sample(sample_shape=torch.Size((n, M)))
            for i in range(1, n + 1):
                s[i, :] = s[i-1, :] + s[i-1, :] * mu * dt + \
                    delta * (s[i-1, :] ** (beta / 2)) * Z[i-1, :] * dt**0.5
        return s
