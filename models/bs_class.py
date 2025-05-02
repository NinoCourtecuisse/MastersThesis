import torch
from torch.distributions import Normal

class Bs(torch.nn.Module):
    def __init__(self, mu, sigma):
        super().__init__()
        """
        Reparametrize to enforce the following constraint: 0 < sigma
        """
        self.mu = torch.nn.Parameter(mu)
        self.l_sigma = torch.nn.Parameter(torch.log(sigma))     # sigma -> log(sigma)
        self.params_names = ['mu', 'sigma']
        self.model_type = 'BS'

    def inv_reparam(self):
        # Inverse the reparametrization.
        return self.mu, torch.exp(self.l_sigma)
    
    def simulate(self, s0, delta_t, T, M):
        with torch.no_grad():
            n = int(torch.round(T / delta_t).item())
            mu, sigma = self.inv_reparam()
            s = torch.zeros((n + 1, M))
            s[0, :] = s0
            Z = Normal(loc=0, scale=1).sample(sample_shape=torch.Size((n, M)))
            log_increments = (mu - 0.5 * sigma**2) * delta_t + sigma * torch.sqrt(delta_t) * Z
            log_price = torch.log(s0) + torch.cumsum(log_increments, dim=0)
            s[1:, :] = torch.exp(log_price)
        return s

    def transition(self, s, s_next, delta_t=1/252):
        """
        Evaluates p_(log(s_next) ; log(s)) over a time-step delta_t. 
        """
        mu, sigma = self.inv_reparam()

        mean = torch.log(s) + (mu - 0.5 * sigma**2) * delta_t
        var = sigma**2 * delta_t
        transition = torch.exp(Normal(loc=mean, scale=torch.sqrt(var)).log_prob(torch.log(s_next)))
        return torch.relu(transition - 1e-6) + 1e-6

    def forward(self, spot_prices, t, window=100, delta_t=1/252):
        start = max(t - window, 0)
        transition_evals = self.transition(spot_prices[start:t], spot_prices[start+1:t+1], delta_t)
        log_likelihood = transition_evals.log().sum()
        return log_likelihood
