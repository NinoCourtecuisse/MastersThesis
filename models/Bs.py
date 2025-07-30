import torch
from torch import distributions as D
from utils.priors import IndependentPrior
from utils.optimization import IndependentTransform

class Bs():
    def __init__(self, dt: float, prior: IndependentPrior):
        self.dt = dt
        self.prior = prior
        self.transform = IndependentTransform(prior)

    def log_transition(self, opt_params, s, s_next):
        # Expects params in optimization parametrization
        params = self.transform.to(opt_params)
        mu = params[:, 0].unsqueeze(1)
        sigma = params[:, 1].unsqueeze(1)
        dt = self.dt

        mean = torch.log(s) + (mu - 0.5 * sigma**2) * dt
        var = sigma**2 * dt
        log_transition = D.Normal(
            loc = mean,
            scale = torch.sqrt(var)
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
            sigma = params[:, 1].unsqueeze(1)

            n = int(torch.round(T / dt).item())
            s = torch.zeros((n + 1, M))
            s[0, :] = s0
            Z = D.Normal(loc=0, scale=1).sample(sample_shape=torch.Size((n, M)))
            log_increments = (mu - 0.5 * sigma**2) * dt + sigma * dt**0.5 * Z
            log_price = torch.log(s0) + torch.cumsum(log_increments, dim=0)
            s[1:, :] = torch.exp(log_price)
        return s
