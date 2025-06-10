import torch
from torch.distributions import Normal

class Bs():
    def __init__(self, dt: float):
        self.dt = dt

    def reparam(self, params):
        mu = params[:, 0]
        sigma = params[:, 1]
        log_sigma = torch.log(sigma)
        new_params = torch.stack([mu, log_sigma], dim = 1)
        return new_params

    def inverse_reparam(self, params):
        mu = params[:, 0]
        sigma = torch.exp(params[:, 1])
        new_params = torch.stack([mu, sigma], dim = 1)
        return new_params

    def log_transition(self, params, s, s_next):
        # Expects params in optimization parametrization
        reparams = self.inverse_reparam(params)
        mu = reparams[:, 0].unsqueeze(1)
        sigma = reparams[:, 1].unsqueeze(1)
        dt = self.dt

        mean = torch.log(s) + (mu - 0.5 * sigma) * dt
        var = sigma**2 * dt
        log_transition = Normal(
            loc = mean,
            scale = torch.sqrt(var)
        ).log_prob(torch.log(s_next))       # TODO: Use log-returns

        return torch.relu(log_transition + 15) - 15

    def ll(self, params: torch.tensor, data: torch.tensor):
        # Expects params in optimization parametrization
        s = data[:-1]
        s_next = data[1:]
        log_transitions = self.log_transition(params, s, s_next)
        ll = torch.sum(log_transitions, dim = 1)
        return ll

    def simulate(self, params, s0, T, M):
        # Expects params in natural parametrization
        with torch.no_grad():
            dt = self.dt
            mu = params[:, 0].unsqueeze(1)
            sigma = params[:, 1].unsqueeze(1)

            n = int(torch.round(T / dt).item())
            s = torch.zeros((n + 1, M))
            s[0, :] = s0
            Z = Normal(loc=0, scale=1).sample(sample_shape=torch.Size((n, M)))
            log_increments = (mu - 0.5 * sigma**2) * dt + sigma * torch.sqrt(dt) * Z
            log_price = torch.log(s0) + torch.cumsum(log_increments, dim=0)
            s[1:, :] = torch.exp(log_price)
        return s
