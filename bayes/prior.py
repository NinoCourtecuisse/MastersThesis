import torch
from torch.distributions import Distribution

class Prior:
    def __init__(self, dists: list[Distribution]):
        """
        Initialize a multidimensional prior.
        Args:
            dists (list): A list of PyTorch distributions, one per dimension.
        """
        self.dists = dists
        self.dim = len(dists)

    def sample(self, n: int = 1) -> torch.Tensor:
        """
        Sample n points from the prior.
        Returns:
            Tensor of shape (n, dim)
        """
        samples = [dist.sample((n,)) for dist in self.dists]
        return torch.stack(samples, dim=-1).squeeze()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probability of batch x.
        Args:
            x (Tensor): shape (n, dim)
        Returns:
            log_prob (Tensor): shape (n,)
        """
        log_probs = [d.log_prob(x[:, i]) for i, d in enumerate(self.dists)]
        return torch.stack(log_probs, dim=-1).sum(dim=-1)
