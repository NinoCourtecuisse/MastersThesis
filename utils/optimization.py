import torch
from torch import distributions as D
from utils.priors import IndependentPrior

class IndependentTransform:
    def __init__(self, prior: IndependentPrior):
        self.transforms = []
        for i, d in enumerate(prior.dists):
            constraint = d.support
            transform = D.transform_to(constraint)
            self.transforms.append(transform)
    
    def to(self, unconstrained_x: torch.Tensor) -> torch.Tensor:
        constrained_x = []
        for i, transform in enumerate(self.transforms):
            constrained_x.append(transform(unconstrained_x[:, i]))
        return torch.stack(constrained_x, dim=-1)
    
    def inv(self, constrained_x: torch.Tensor) -> torch.Tensor:
        unconstrained_x = []
        for i, transform in enumerate(self.transforms):
            unconstrained_x.append(transform.inv(constrained_x[:, i]))
        return torch.stack(unconstrained_x, dim=-1)
