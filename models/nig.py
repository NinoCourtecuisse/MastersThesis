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
    def __init__(self, mu, alpha, beta, delta):
        """
        Reparametrize to enforce: alpha > 0, -1 < beta / alpha < 1, delta > 0
        """
        
        psi = beta / alpha
        params = torch.nn.ParameterDict({
            'mu': torch.nn.Parameter(mu),
            'log_alpha': torch.nn.Parameter(torch.log(alpha)),
            'atanh_psi': torch.nn.Parameter(torch.atanh(psi)),
            'log_delta': torch.nn.Parameter(torch.log(delta)),
        })
        #self.mu = mu
        params_names = ['mu', 'alpha', 'beta', 'delta']
        super().__init__(params, params_names)

        self.model_type = 'NIG'

    def get_params(self):
        mu = self.params["mu"].detach()
        alpha = torch.exp(self.params["log_alpha"].detach())
        psi = torch.tanh(self.params["atanh_psi"].detach())
        delta = torch.exp(self.params['log_delta'].detach())
        return mu, alpha, alpha * psi, delta

    def inv_reparam(self):
        mu = self.params["mu"]
        alpha = torch.exp(self.params["log_alpha"])
        psi = torch.tanh(self.params["atanh_psi"])
        delta = torch.exp(self.params['log_delta'])
        return mu, alpha, alpha * psi, delta

    def get_moments(self, t):
        """
        Return the first two moments of x_t = log(s_t / s_0)
        """
        mu, alpha, beta, delta = self.inv_reparam()
        gamma = torch.sqrt(alpha**2 - beta**2)
        mean = mu + delta * beta / gamma
        var = delta * alpha**2 / gamma**3
        return t * mean, t * torch.sqrt(var)

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
    
    def forward(self, spot_prices, t, delta_t, window):
        start = max(t - window, 0)
        #self.mu = torch.mean(torch.log(spot_prices[start+1:t+1] / spot_prices[start:t])) / delta_t \
        #            - delta * beta / torch.sqrt(alpha**2 - beta**2)

        transition_evals = self.transition(spot_prices[start:t], spot_prices[start+1:t+1], delta_t)
        #log_likelihood = transition_evals.log().sum()
        log_likelihood = transition_evals.sum()

        return log_likelihood

