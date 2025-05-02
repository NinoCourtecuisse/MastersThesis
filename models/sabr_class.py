import torch
from torch.distributions import Normal, Poisson, Chi2, LogNormal
import math

class Sabr(torch.nn.Module):
    def __init__(self, mu, beta, sigma, rho, delta_0):
        super().__init__()
        """
        Reparametrize to enforce the following constraints:
        0 < beta < 2 ; 0 < theta ; 0 < sigma ; -1 < rho < 1
        """
        #self.mu = torch.nn.Parameter(mu)
        self.mu = mu
        self.at_beta = torch.nn.Parameter(torch.atanh(beta - 1))     # theta -> log(theta)
        self.l_sigma = torch.nn.Parameter(torch.log(sigma))          # sigma -> log(sigma)
        self.at_rho = torch.nn.Parameter(torch.atanh(rho))      # rho -> atanh(rho)
        
        self.delta_0 = delta_0
        self.params_names = ['mu', 'beta', 'sigma', 'rho', 'delta_0']
        self.model_type = 'SV'

    def inv_reparam(self):
        # Inverse the reparametrization.
        return self.mu, torch.tanh(self.at_beta) + 1, torch.exp(self.l_sigma), torch.tanh(self.at_rho), self.delta_0
    
    def local_var(self, t, delta, s):
        mu, beta, sigma, rho, _ = self.inv_reparam()
        return torch.exp((2 - beta) * mu * t) * delta**2 * s**(beta - 2)
    
    def wu_transition(self, s, v, s_next, v_next, delta_t=1/252):
        # Joint transition of (f, v)
        mu, beta, sigma, rho, _ = self.inv_reparam()
        f = s
        F = s_next * torch.exp(-mu * delta_t)
        u = (f**(1 - beta / 2) - F**(1 - beta / 2)) / (v * (1 - beta/2) * torch.sqrt(delta_t))
        vv = torch.log(v / v_next) / (sigma * torch.sqrt(delta_t))
        a11 = - beta / 2 * F**(beta / 2 - 1) * v_next / (sigma * (u - rho * sigma))
        a10 = u**2 * sigma - rho * u * sigma**2
        joint = 1 / (sigma * delta_t * F**(beta / 2) * v_next**2) * \
                    (1 + sigma * torch.sqrt(delta_t) / (2 * (-1 + rho**2))  * (a11 + a10)) \
                    / (2 * math.pi * torch.sqrt(1 - rho**2)) * \
                    torch.exp(-(u**2 - 2 * rho * u * sigma + sigma**2) / (2 * (1 - rho**2)))
        return joint
    
    def variance_path(self, spot_prices, delta_t):
        mu, beta, sigma, rho, delta_0 = self.inv_reparam()
        log_spot_diff = torch.log(spot_prices[1:]) - torch.log(spot_prices[:-1])
        delta_path = torch.zeros_like(spot_prices)
        delta_path[0] = delta_0
        T = len(spot_prices)
        for t in range(1, T):
            delta_prev = delta_path[t-1].clone()
            var_prev = torch.relu(self.local_var(t * delta_t, delta_prev, spot_prices[t-1]) - 1e-2) + 1e-2
            mu_tilde = torch.log(delta_prev) - 0.5 * sigma**2 * delta_t + rho * sigma**2 * \
                (log_spot_diff[t-1] - (mu - 0.5 * var_prev) * delta_t) / var_prev
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
    
    def forward(self, spot_prices, t, window, delta_t=1/252, update_v0=False):
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
