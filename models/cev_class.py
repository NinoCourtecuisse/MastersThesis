from models.model_class import Model

import torch
from torch.distributions import Normal

class Cev(Model):
    def __init__(self, mu, delta, beta):
        """
        Reparametrize to enforce: delta > 0, beta > 0
        """
        params = torch.nn.ParameterList([
            torch.nn.Parameter(mu),
            torch.nn.Parameter(torch.log(delta)),
            torch.nn.Parameter(torch.log(beta))
        ])
        params_names = ['mu', 'delta', 'beta']
        super().__init__(params, params_names)

        self.model_type = 'CEV'

    def inv_reparam(self):
        # Inverse the reparametrization.
        return self.params[0], torch.exp(self.params[1]), torch.exp(self.params[2])

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

    def transition(self, s, s_next, delta_t=1/252):
        """
        Evaluates p_(log(s_next) ; log(s)) over a time-step delta_t. 
        """
        mu, delta, beta = self.inv_reparam()

        C = torch.tensor(10.0)
        tmp = torch.cat([C**2 * torch.exp((beta - 2) * mu * delta_t) * torch.ones_like(s.view(-1, 1)), s.view(-1, 1)**(beta - 2)], dim=1)
        local_var = delta**2 * torch.min(tmp, dim=1)[0]

        mean = torch.log(s) + (mu - 0.5 * local_var) * delta_t
        var = local_var * delta_t
        transition = torch.exp(Normal(loc=mean, scale=torch.sqrt(var)).log_prob(torch.log(s_next)))
        return torch.relu(transition - 1e-6) + 1e-6

    def forward(self, spot_prices, t, delta_t, window=100):
        start = max(t - window, 0)
        transition_evals = self.transition(spot_prices[start:t], spot_prices[start+1:t+1], delta_t)
        log_likelihood = transition_evals.log().sum()
        return log_likelihood
