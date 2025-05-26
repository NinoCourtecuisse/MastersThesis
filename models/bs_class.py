from models.model_class import Model

import torch
from torch.distributions import Normal

class Bs(Model):
    def __init__(self, mu, sigma):
        """
        Reparametrize to enforce: sigma > 0
        """
        params = torch.nn.ParameterDict({
            'mu': torch.nn.Parameter(mu),
            'log_sigma': torch.nn.Parameter(torch.log(sigma))
        })
        params_names = ['mu', 'sigma']
        super().__init__(params, params_names)

        self.model_type = 'BS'

    def get_params(self):
        return self.params['mu'].detach(), torch.exp(self.params['log_sigma'].detach())

    def inv_reparam(self):
        # Inverse the reparametrization.
        return self.params['mu'], torch.exp(self.params['log_sigma'])
    
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

    def transition(self, s, s_next, delta_t):
        """
        Evaluates p_(log(s_next) ; log(s)) over a time-step delta_t. 
        """
        mu, sigma = self.inv_reparam()

        mean = torch.log(s) + (mu - 0.5 * sigma**2) * delta_t
        var = sigma**2 * delta_t
        transition = Normal(loc=mean, scale=torch.sqrt(var)).log_prob(torch.log(s_next))
        return torch.relu(transition + 15) - 15

    def forward(self, spot_prices, t, delta_t, beta, window=None):
        if window:
            start = max(t - window, 0)
        else:
            start = 0

        #self.params['mu'] = torch.mean((spot_prices[start+1:t+1] - spot_prices[start:t]) / spot_prices[start:t])
        transition_evals = self.transition(spot_prices[start:t], spot_prices[start+1:t+1], delta_t)
        #log_likelihood = transition_evals.log().sum()
        decay = beta**(torch.arange(t - 1, start - 1, -1))
        log_likelihood = decay.inner(transition_evals)
        return log_likelihood
