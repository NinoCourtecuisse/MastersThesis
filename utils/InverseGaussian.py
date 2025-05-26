import torch
from torch.distributions import Normal
from math import pi

class InverseGaussian(torch.distributions.Distribution):
    def __init__(self, mu, lam):
        self.mu = mu
        self.lam = lam

    def _get_params(self):
        return self.mu, self.lam

    def log_prob(self, x):
        mu, lam = self._get_params()
        return 0.5 * torch.log(lam / (2 * pi * x**3)) \
                - lam * (x - mu)**2 / (2 * mu**2 * x)
    
    def sample(self, sample_shape):
        mu, lam = self._get_params()

        # Michael-Schucany-Haas method
        v = torch.randn(sample_shape)**2
        v1 = mu + mu**2 * v / (2 * lam) \
            - mu / (2 * lam) * torch.sqrt(4 * mu * lam * v + mu**2 * v**2)
        z = torch.rand(sample_shape)

        v_sample = torch.where(z <= mu / (mu + v1), v1, mu**2 / v1)
        return torch.relu(v_sample - 1e-10) + 1e-10
