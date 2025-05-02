import torch
from torch.distributions import Normal

class Cev(torch.nn.Module):
    def __init__(self, mu, delta, beta):
        super().__init__()
        """
        Reparametrize to enforce the following constraint: 
        0 < delta, 0 < beta
        """
        self.mu = torch.nn.Parameter(mu)
        self.l_delta = torch.nn.Parameter(torch.log(delta))     # sigma -> log(sigma)
        self.l_beta = torch.nn.Parameter(torch.log(beta))       # beta -> log(beta)
        self.params_names = ['mu', 'delta', 'beta']
        self.model_type = 'CEV'

    def inv_reparam(self):
        # Inverse the reparametrization.
        return self.mu, torch.exp(self.l_delta), torch.exp(self.l_beta)
    
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
    
    def forward(self, spot_prices, t, window=100, delta_t=1/252):
        start = max(t - window, 0)
        transition_evals = self.transition(spot_prices[start:t], spot_prices[start+1:t+1], delta_t)
        log_likelihood = transition_evals.log().sum()
        return log_likelihood
