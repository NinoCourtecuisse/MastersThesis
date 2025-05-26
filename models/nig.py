import torch
from models.model_class import Model
from distributions.NormalInverseGaussian import NormalInverseGaussian
from distributions.InverseGaussian import InverseGaussian
from torch.distributions import Normal
from torch.special import modified_bessel_k1 as k1
from torch.special import scaled_modified_bessel_k1 as k1_scaled
import math

from torch.autograd import Function
from scipy.special import k0e, k1e

class ScaledBesselK1(Function):
    @staticmethod
    def forward(ctx, input):
        x_np = input.detach().cpu().numpy()
        scaled_k1 = k1e(x_np)
        ctx.save_for_backward(input)
        return torch.tensor(scaled_k1, dtype=input.dtype, device=input.device)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        input = input.detach()

        # Derivative of e^x * K1(x):
        # d/dx [e^x K1(x)] = e^x (K1(x) + K1'(x))
        # We use: K1'(x) = -K0(x) - K1(x)/x
        x_np = input.cpu().numpy()
        d_scaled_k1 = -k0e(x_np) + k1e(x_np) * (1 - 1 / x_np)

        grad_input = grad_output * torch.tensor(d_scaled_k1, dtype=input.dtype, device=input.device)
        return grad_input
    
def torch_k1e(x):
    return ScaledBesselK1.apply(x)


class Nig(Model):
    def __init__(self, mu, sigma, xi, neta):
        """
        Reparametrize to enforce: sigma, neta > 0
        """
        params = torch.nn.ParameterDict({
            'mu': torch.nn.Parameter(mu),
            'log_sigma': torch.nn.Parameter(torch.log(sigma)),
            'xi': torch.nn.Parameter(xi),
            'log_neta': torch.nn.Parameter(torch.log(neta)),
        })
        params_names = ['mu', 'sigma', 'xi', 'neta']
        super().__init__(params, params_names)

        self.model_type = 'NIG'

    def get_params(self):
        mu = self.params["mu"].detach()
        sigma = torch.exp(self.params["log_sigma"].detach())
        xi = self.params["xi"].detach()
        neta = torch.exp(self.params['log_neta'].detach())
        return mu, sigma, xi, neta

    def inv_reparam(self):
        "Inverse reparametrization to the original."
        mu = self.params["mu"]
        sigma = torch.exp(self.params["log_sigma"])
        xi = self.params["xi"]
        neta = torch.exp(self.params['log_neta'])
        
        sigma_tilde = sigma / torch.sqrt(1 + neta * xi**2)

        mu_tilde = -sigma_tilde * xi + mu
        alpha_tilde = torch.sqrt(1/neta + xi**2) / sigma_tilde
        beta_tilde = xi / sigma_tilde
        delta_tilde = sigma_tilde / torch.sqrt(neta)
        return mu_tilde, alpha_tilde, beta_tilde, delta_tilde

    def get_moments(self):
        """
        Return the first two moments of x_t = log(s_t / s_0) at t=1
        """
        mu, alpha, beta, delta = self.inv_reparam()
        gamma = torch.sqrt(alpha**2 - beta**2)

        mean = mu + delta * beta / gamma
        var = delta * alpha**2 / gamma**3
        skew = 3 * beta / (alpha * torch.sqrt(delta * gamma))
        ekurt = 3 * (1 + 4 * beta**2 / alpha**2) / (delta * gamma)
        return mean, torch.sqrt(var), skew, ekurt

    def simulate(self, s0, dt, T, M=1):
        with torch.no_grad():
            mu, alpha, beta, delta = self.inv_reparam()
            n = int(torch.round(T / dt).item())

            # Generate M paths from IG process
            dI = InverseGaussian(
                mu = delta * dt / torch.sqrt(alpha**2 - beta**2),
                lam = (delta * dt)**2
            ).sample(sample_shape=(n - 1, M))
            I = torch.cumsum(dI, dim=0)

            # Generate M paths from NIG process
            Z = Normal(
                loc=torch.tensor(0.0),
                scale=torch.tensor(1.0)
            ).sample(sample_shape=(n - 1, M))
            W_It = torch.cumsum(torch.sqrt(dI) * Z, dim=0)
            X = mu * torch.linspace(dt, T, n - 1).unsqueeze(1) + beta * I + W_It

            s = s0 * torch.ones((n, M))
            s[1:, :] = s0 * torch.exp(X)
        return s
    
    def transition(self, s, s_next, delta_t):
        mu, alpha, beta, delta = self.inv_reparam()
        #if torch.isnan(alpha.log()).any():
        #    print('aie...')
        log_return = torch.log(s_next / s)
        mu_t = delta_t * mu
        delta_time = delta * delta_t
        tmp = alpha * torch.sqrt(delta_time**2 + (log_return - mu_t)**2)
        log_transition = torch.log(alpha * delta_time) \
                    + torch.log(torch_k1e(alpha * torch.sqrt(delta_time**2 + (log_return - mu_t)**2))).float() \
                    - alpha * torch.sqrt(delta_time**2 + (log_return - mu_t)**2) \
                    - torch.log(math.pi * torch.sqrt(delta_time**2 + (log_return - mu_t)**2)) \
                    + delta_time * torch.sqrt(alpha**2 - beta**2) + beta * (log_return - mu_t)
        #transition = torch.exp(log_transition)
        #transition = torch.exp(NormalInverseGaussian(delta_t * mu, alpha, beta, delta * delta_t).log_prob(log_return))
        #if torch.isnan(transition.log()).any():
        #    print('aie...')
        return torch.relu(log_transition + 15) - 15
        return torch.relu(transition - 1e-6) + 1e-6
    
    def forward(self, spot_prices, t, delta_t, beta, window=None):
        if window:
            start = max(t - window, 0)
        else:
            start = 0
        #self.params["mu"] = torch.mean(torch.log(spot_prices[start+1:t+1] / spot_prices[start:t])) / delta_t

        transition_evals = self.transition(spot_prices[start:t], spot_prices[start+1:t+1], delta_t)
        #log_likelihood = transition_evals.log().sum()
        #log_likelihood = transition_evals.sum()
        decay = beta**(torch.arange(t - 1, start - 1, -1))
        log_likelihood = decay.inner(transition_evals)

        return log_likelihood

