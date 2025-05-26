import torch
import math
from utils.Bessel import torch_k1e
from utils.InverseGaussian import InverseGaussian

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
        epsilon = torch.randn(size=sample_shape)
        x = mu + beta * v + torch.sqrt(v) * epsilon
        return x

    def log_prob(self, x):
        mu, alpha, beta, delta = self._get_params()
        return torch.log(alpha * delta) \
                + torch.log(torch_k1e(alpha * torch.sqrt(delta**2 + (x - mu)**2))).float() \
                - alpha * torch.sqrt(delta**2 + (x - mu)**2) \
                - torch.log(math.pi * torch.sqrt(delta**2 + (x - mu)**2)) \
                + delta * torch.sqrt(alpha**2 - beta**2) + beta * (x - mu)
