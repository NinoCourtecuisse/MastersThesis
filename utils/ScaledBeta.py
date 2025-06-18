import torch
from torch import distributions as D

class ScaledBeta:
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
        # Map x back to [0,1]
        u = (x - self.low) / (self.high - self.low)
        # Clamp to avoid numerical issues if exactly on boundaries
        u = u.clamp(1e-6, 1.0 - 1e-6)
        return self.beta_dist.log_prob(u) - torch.log(self.high - self.low)

    @property
    def support(self):
        # To match Uniform convention
        return torch.distributions.constraints.interval(self.low, self.high)
