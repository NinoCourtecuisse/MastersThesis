import torch
from torch import distributions as D
from math import pi
from utils.special_functions import torch_k1e

class InverseGaussian(D.Distribution):
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

class NormalInverseGaussian(D.Distribution):
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
                - torch.log(pi * torch.sqrt(delta**2 + (x - mu)**2)) \
                + delta * torch.sqrt(alpha**2 - beta**2) + beta * (x - mu)

class ScaledBeta(D.Distribution):
    def __init__(self, alpha, beta, low, high):
        self.beta_dist = D.Beta(alpha, beta)
        self.low = low
        self.high = high

    def sample(self, sample_shape):
        u = self.beta_dist.sample(sample_shape)
        return self.low + (self.high - self.low) * u

    def rsample(self, sample_shape):
        u = self.beta_dist.rsample(sample_shape)
        return self.low + (self.high - self.low) * u

    def log_prob(self, x):
        u = (x - self.low) / (self.high - self.low)
        # Clamp to avoid numerical issues if exactly on boundaries
        u = u.clamp(1e-10, 1.0 - 1e-10)
        return self.beta_dist.log_prob(u) - torch.log(self.high - self.low)

    @property
    def support(self):
        return torch.distributions.constraints.interval(self.low, self.high)
