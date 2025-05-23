import torch
import math
from torch.special import modified_bessel_k1 as k1
from distributions.InverseGaussian import InverseGaussian
from torch.distributions import Normal

class NormalInverseGaussian(torch.distributions.Distribution):
    def __init__(self, mu, alpha, beta, delta):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def _get_params(self):
        return self.mu, self.alpha, self.beta, self.delta
    
    def sample(self, sample_shape):
        mu, alpha, beta, delta = self._get_params()

        v = InverseGaussian(
            delta / torch.sqrt(alpha**2 - beta**2),
            delta**2
        ).sample(sample_shape)
        epsilon = Normal(
            loc=torch.tensor(0.0),
            scale=torch.tensor(1.0)
        ).sample(sample_shape)

        x = mu + beta * v + torch.sqrt(v) * epsilon
        return x

    def log_prob(self, x):
        mu, alpha, beta, delta = self._get_params()
        return torch.log(alpha * delta * k1(alpha * torch.sqrt(delta**2 + (x - mu)**2))) \
                - torch.log(math.pi * torch.sqrt(delta**2 + (x - mu)**2)) \
                + delta * torch.sqrt(alpha**2 - beta**2) + beta * (x - mu)
