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

    @classmethod
    def from_moments(cls, mu, sigma, gamma_1, gamma_2):
        mu_, alpha, beta, delta = cls.reparametrize(mu, sigma, gamma_1, gamma_2)
        return cls(mu_, alpha, beta, delta)

    @staticmethod
    def reparametrize(mu, sigma, gamma_1, gamma_2):
        if not torch.all(gamma_2 > 5 * gamma_1**2 / 3):
            raise ValueError(
                f'The skewness (gamma_1) and excess kurtosis (gamma_2) must satisfy: '
                f'3 * gamma_2 > 5 * gamma_1**2, input: {mu, sigma, gamma_1, gamma_2}'
            )

        xi, eta = NormalInverseGaussian._phi_1(gamma_1, gamma_2)
        mu_, alpha, beta, delta = NormalInverseGaussian._phi_2(mu, sigma, xi, eta)
        return mu_, alpha, beta, delta

    @staticmethod
    def _phi_1(gamma_1, gamma_2):
        xi = gamma_1 / torch.sqrt((gamma_2 - 4 * gamma_1**2 / 3) * (gamma_2 - 5 * gamma_1**2 / 3))
        eta = (gamma_2 - 4 * gamma_1**2 / 3) / 3
        return xi, eta

    @staticmethod
    def _phi_2(mu, sigma, xi, eta):
        sigma_tilde = sigma / torch.sqrt(1 + eta * xi**2)

        mu_ = - sigma_tilde * xi + mu
        alpha = torch.sqrt(1/eta + xi**2) / sigma_tilde
        beta = xi / sigma_tilde
        delta = sigma_tilde / torch.sqrt(eta)
        return mu_, alpha, beta, delta

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

        if isinstance(low, float):
            low = torch.tensor(low)
        if isinstance(high, float):
            high = torch.tensor(high)
        self.low = low
        self.high = high

    def sample(self, sample_shape=torch.Size([])):
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
