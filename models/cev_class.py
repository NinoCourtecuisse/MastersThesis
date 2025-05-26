from models.model_class import Model

import torch
from torch.distributions import Normal

class Cev(Model):
    def __init__(self, mu, delta, beta):
        """
        Reparametrize to enforce: delta > 0, beta > 0
        """
        params = torch.nn.ParameterDict({
            'mu': torch.nn.Parameter(mu),
            'log_delta': torch.nn.Parameter(torch.log(delta)),
            'log_beta': torch.nn.Parameter(torch.log(beta))
        })
        params_names = ['mu', 'delta', 'beta']
        super().__init__(params, params_names)

        self.model_type = 'CEV'

    def inv_reparam(self):
        # Inverse the reparametrization.
        return self.params['mu'], torch.exp(self.params['log_delta']), \
                torch.exp(self.params['log_beta'])
    
    def get_params(self):
        return self.params['mu'].detach(), torch.exp(self.params['log_delta'].detach()), \
                torch.exp(self.params['log_beta'].detach())

    def simulate(self, s0, delta_t, T, M):
        with torch.no_grad():
            n = int(torch.round(T / delta_t).item())
            mu, delta, beta = self.inv_reparam()
            Z = Normal(loc=0, scale=1).sample(sample_shape=torch.Size((n, M)))
            s = torch.zeros((n + 1, M))
            s[0, :] = s0
            for i in range(1, n + 1):
                s[i, :] = s[i-1, :] + s[i-1, :] * mu * delta_t + \
                    delta * (s[i-1, :] ** (beta / 2)) * Z[i-1, :] * torch.sqrt(delta_t)
        return s

    def transition(self, s, s_next, delta_t):
        """
        Evaluates p_(log(s_next) ; log(s)) over a time-step delta_t. 
        """
        mu, delta, beta = self.inv_reparam()

        C = torch.tensor(10.0)
        tmp = torch.cat([C**2 * torch.exp((beta - 2) * mu * delta_t) * torch.ones_like(s.view(-1, 1)), s.view(-1, 1)**(beta - 2)], dim=1)
        local_var = delta**2 * torch.min(tmp, dim=1)[0]

        mean = torch.log(s) + (mu - 0.5 * local_var) * delta_t
        var = local_var * delta_t
        transition = Normal(loc=mean, scale=torch.sqrt(var)).log_prob(torch.log(s_next))
        return torch.relu(transition + 15) - 15

    def forward(self, spot_prices, t, delta_t, decay_coef, window=None):
        if window:
            start = max(t - window, 0)
        else:
            start = 0
        #self.params['mu'] = torch.mean((spot_prices[start+1:t+1] - spot_prices[start:t]) / spot_prices[start:t])
        transition_evals = self.transition(spot_prices[start:t], spot_prices[start+1:t+1], delta_t)
        #log_likelihood = transition_evals.log().sum()
        decay = decay_coef**(torch.arange(t - 1, start - 1, -1))
        log_likelihood = decay.inner(transition_evals)
        return log_likelihood
