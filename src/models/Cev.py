import torch
from torch import distributions as D
from src.utils.priors import IndependentPrior
from src.utils.optimization import IndependentTransform, CevTransform

class Cev():
    def __init__(self, dt: float, prior: IndependentPrior):
        self.dt = dt
        self.prior = prior
        self.transform = CevTransform(prior)

    def log_transition(self, opt_params, s, s_next):
        # Expects params in optimization parametrization
        params = self.transform.to(opt_params)
        mu = params[:, 0].unsqueeze(1)
        delta = params[:, 1].unsqueeze(1)
        beta = params[:, 2].unsqueeze(1)
        dt = self.dt

        C = torch.tensor(10**3)
        tmp1 = C**2 * torch.exp((beta - 2) * mu * dt) * torch.ones(size=(len(mu), len(s)))
        tmp2 = s[None, :]**(beta - 2)
        local_var = delta**2 * torch.minimum(tmp1, tmp2)
        #local_var = delta**2 * s[None, :]**(beta - 2)

        mean = torch.log(s) + (mu - 0.5 * local_var) * dt
        var = local_var * dt
        log_transition = D.Normal(
            loc=mean,
            scale=torch.sqrt(var)
        ).log_prob(torch.log(s_next))
        return log_transition

    def ll(self, opt_params, data):
        # Expects params in optimization parametrization
        s = data[:-1]
        s_next = data[1:]
        log_transitions = self.log_transition(opt_params, s, s_next)
        ll = torch.sum(log_transitions, dim = 1)
        return ll

    def lpost(self, opt_params, data):
        llik = self.ll(opt_params, data)
        lprior = self.prior.log_prob(self.transform.to(opt_params))
        return llik + lprior

    def simulate(self, params, s0, T, M):
        # Expects params in natural parametrization
        with torch.no_grad():
            dt = self.dt
            mu = params[:, 0].unsqueeze(1)
            delta = params[:, 1].unsqueeze(1)
            beta = params[:, 2].unsqueeze(1)

            n = int(torch.round(T / dt).item())
            s = torch.zeros((n + 1, M))
            s[0, :] = s0
            Z = D.Normal(loc=0, scale=1).sample(sample_shape=torch.Size((n, M)))
            for i in range(1, n + 1):
                s[i, :] = s[i-1, :] + s[i-1, :] * mu * dt + \
                    delta * (s[i-1, :] ** (beta / 2)) * Z[i-1, :] * dt**0.5
        return s
