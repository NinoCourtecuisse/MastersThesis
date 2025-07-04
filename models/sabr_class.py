from models.model_class import Model

import torch
from torch.distributions import Normal

class Sabr(Model):
    def __init__(self, mu, beta, sigma, rho, delta_0):
        """
        Reparametrize to enforce: 0 < beta < 2 ; theta > 0 ; sigma > 0 ; -1 < rho < 1
        """
        params = torch.nn.ParameterList([
            torch.nn.Parameter(torch.atanh(beta - 1)),
            torch.nn.Parameter(torch.log(sigma)),
            torch.nn.Parameter(torch.atanh(rho))
        ])
        self.mu = mu
        self.delta_0 = delta_0

        params_names = ['mu', 'beta', 'sigma', 'rho', 'delta_0']
        super().__init__(params, params_names)
        self.model_type = 'SV'

    def inv_reparam(self):
        # Inverse the reparametrization.
        return self.mu, torch.tanh(self.params[0]) + 1, torch.exp(self.params[1]), torch.tanh(self.params[2]), self.delta_0
    
    def get_params(self):
        return [torch.tanh(self.params[0]) + 1, torch.exp(self.params[1]), torch.tanh(self.params[2])]
    
    def local_var(self, t, delta, s):
        mu, beta, sigma, rho, _ = self.inv_reparam()
        #return torch.exp((2 - beta) * mu * t) * delta**2 * s**(beta - 2)
        return delta**2 * s**(beta - 2)
    
    def variance_path(self, spot_prices, delta_t):
        mu, beta, sigma, rho, delta_0 = self.inv_reparam()
        log_spot_diff = torch.log(spot_prices[1:]) - torch.log(spot_prices[:-1])
        delta_path = torch.zeros_like(spot_prices)
        delta_path[0] = delta_0
        T = len(spot_prices)
        for t in range(1, T):
            delta_prev = delta_path[t-1].clone()
            var_prev = torch.relu(self.local_var(t * delta_t, delta_prev, spot_prices[t-1]) - 1e-2) + 1e-2
            mu_tilde = torch.log(delta_prev) - 0.5 * sigma**2 * delta_t + rho * sigma * \
                (log_spot_diff[t-1] - (mu - 0.5 * var_prev) * delta_t) / torch.sqrt(var_prev)
            sigma_tilde = torch.sqrt((1 - rho**2) * sigma**2 * delta_t)
            raw_delta = torch.exp(mu_tilde + 0.5 * sigma_tilde**2)
            delta_path[t] = raw_delta
        var_path = self.local_var(torch.arange(0, T) * delta_t, delta_path, spot_prices)
        return var_path, delta_path
    
    def euler_transition(self, s, v, s_next, delta_t):
        mu, beta, sigma, rho, delta_0 = self.inv_reparam()

        mean = torch.log(s) + (mu - 0.5 * v) * delta_t
        var = v * delta_t
        transition = torch.exp(Normal(loc=mean, scale=torch.sqrt(var)).log_prob(torch.log(s_next)))
        return torch.relu(transition - 1e-6) + 1e-6
    
    def forward(self, spot_prices, t, delta_t, window, update_v0=False):
        start = max(t - window, 0)
        self.mu = torch.mean((spot_prices[start+1:t+1] - spot_prices[start:t]) / spot_prices[start:t])

        var_path, delta_path = self.variance_path(spot_prices[start:t], delta_t)

        transition_evals = self.euler_transition(spot_prices[start:t], var_path, spot_prices[start+1:t+1], delta_t)
        log_likelihood = transition_evals.log().sum()
        if update_v0:
            self.delta_0 = delta_path[1].detach().clone()
        #decay = neta**(torch.arange(t - 1, -1, -1))
        #log_likelihood = decay.inner(transition_evals.log())
        return log_likelihood
