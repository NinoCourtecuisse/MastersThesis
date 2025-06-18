import torch
from models.model_class import Model
from utils.NormalInverseGaussian import NormalInverseGaussian
from utils.InverseGaussian import InverseGaussian

class Nig(Model):
    def __init__(self, mu, sigma, xi, eta):
        params = torch.nn.ParameterDict({
            'mu': torch.nn.Parameter(mu),
            'log_sigma': torch.nn.Parameter(torch.log(sigma)),
            'xi': torch.nn.Parameter(xi),
            'log_eta': torch.nn.Parameter(torch.log(eta)),
        })
        super().__init__(params)
        self.model_type = 'NIG'

    def get_params(self):
        params = {
            'mu': self.params["mu"].detach(),
            'sigma': torch.exp(self.params["log_sigma"].detach()),
            'xi': self.params["xi"].detach(),
            'eta': torch.exp(self.params['log_eta'].detach())
        }
        return params

    def _inv_reparam(self):
        "Inverse reparametrization to the original."
        mu = self.params["mu"]
        sigma = torch.exp(self.params["log_sigma"])
        xi = self.params["xi"]
        eta = torch.exp(self.params['log_eta'])
        
        sigma_tilde = sigma / torch.sqrt(1 + eta * xi**2)

        mu_tilde = -sigma_tilde * xi + mu
        alpha_tilde = torch.sqrt(1/eta + xi**2) / sigma_tilde
        beta_tilde = xi / sigma_tilde
        delta_tilde = sigma_tilde / torch.sqrt(eta)
        return mu_tilde, alpha_tilde, beta_tilde, delta_tilde

    def get_moments(self):
        """
        Return the first two moments of x_t = log(s_t / s_0) at t=1
        """
        mu, alpha, beta, delta = self._inv_reparam()
        gamma = torch.sqrt(alpha**2 - beta**2)

        mean = mu + delta * beta / gamma
        var = delta * alpha**2 / gamma**3
        skew = 3 * beta / (alpha * torch.sqrt(delta * gamma))
        ekurt = 3 * (1 + 4 * beta**2 / alpha**2) / (delta * gamma)
        return mean, torch.sqrt(var), skew, ekurt

    def simulate(self, s0, dt, T, M=1):
        with torch.no_grad():
            mu, alpha, beta, delta = self._inv_reparam()
            n = int(torch.round(T / dt).item())

            # Generate M paths from IG process
            dI = InverseGaussian(
                mu = delta * dt / torch.sqrt(alpha**2 - beta**2),
                lam = (delta * dt)**2
            ).sample(sample_shape=(n - 1, M))
            I = torch.cumsum(dI, dim=0)

            # Generate M paths from NIG process
            Z = torch.randn(size=(n - 1, M))
            W_It = torch.cumsum(torch.sqrt(dI) * Z, dim=0)
            X = mu * torch.linspace(dt, T, n - 1).unsqueeze(1) + beta * I + W_It

            s = s0 * torch.ones((n, M))
            s[1:, :] = s0 * torch.exp(X)
        return s
    
    def log_transition(self, s, s_next, delta_t):
        mu, alpha, beta, delta = self._inv_reparam()

        log_return = torch.log(s_next / s)
        log_transition = NormalInverseGaussian(delta_t * mu, alpha, beta, delta * delta_t).log_prob(log_return)
        return torch.relu(log_transition + 15) - 15     # Clamp the transition at exp(-15)
    
    def forward(self, spot_prices, t, delta_t, decay_coef, window=None):
        if window:
            start = max(t - window, 0)
        else:
            start = 0

        transition_evals = self.log_transition(spot_prices[start:t], spot_prices[start+1:t+1], delta_t)
        decay = decay_coef**(torch.arange(t - 1, start - 1, -1))
        log_likelihood = decay.inner(transition_evals)
        return log_likelihood
