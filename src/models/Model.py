from abc import ABC, abstractmethod

import torch

from src.utils.priors import Prior

class Model(ABC):
    """
    Base class for all our model classes.

    Args:
        dt (float or torch.Tensor): Represents the time-step Δ over which to compute the transition densities (e.g. Δ = 1 / 252)
        prior (Prior): Prior distribution on the parameters, which also defines a transform to perform unconstrained optimization.
    """
    def __init__(self, dt:float|torch.Tensor, prior:Prior):
        if isinstance(dt, float):
            self.dt = torch.tensor(dt)
        else:
            self.dt = dt
        self.prior = prior
        self.transform = prior.transform

    @abstractmethod
    def log_transition(self, u_params:torch.Tensor, s:torch.Tensor, s_next:torch.Tensor) -> torch.Tensor:
        """
        Evaluates the model's log-transition density, over a time-step Δ:
            log p(s(t), s(t+Δ) | params)

        This method is vectorized over both parameter sets and initial prices:
            Computes log p(s(t), s(t+Δ) | params_i) for i=1, ..., N and t=1, ..., T.

        Args:
            u_params (torch.Tensor): Tensor of shape (N, D), containing N sets of
                                    unconstrained parameters.
            s (torch.Tensor): Tensor of shape (T,), the initial price sequence s(t).
            s_next (torch.Tensor): Tensor of shape (T,), the next state sequence s(t+Δ).

        Returns:
            torch.Tensor: Tensor of shape (N, T), where each entry (i, t) is the log
                      transition density for params_i starting at s(t) and evaluated at s(t+Δ).
        """
        pass

    def ll(self, u_params:torch.Tensor, data:torch.Tensor) -> torch.Tensor:
        """
        Evaluates the model's log-likelihood.

        Args:
            u_params (torch.Tensor): Tensor of shape (N, D), containing N sets of
                                    unconstrained parameters.
            data (torch.Tensor): Tensor of shape (T+1,), the price data over which 
                                to compute the likelihood.

        Returns:
            torch.Tensor: Tensor of shape (N,), where each entry i is the log-likelihood
                        of parameter params_i.
        """
        s = data[:-1]
        s_next = data[1:]
        log_transitions = self.log_transition(u_params, s, s_next)
        ll = torch.sum(log_transitions, dim = 1)
        return ll

    def lpost(self, u_params:torch.Tensor, data:torch.Tensor) -> torch.Tensor:
        """
        Evaluates the model's unnormalized log-posterior.

        Args:
            u_params (torch.Tensor): Tensor of shape (N, D), containing N sets of
                                    unconstrained parameters.
            data (torch.Tensor): Tensor of shape (T+1,), the price data over which 
                                to compute the likelihood.

        Returns:
            torch.Tensor: Tensor of shape (N,), where each entry i is the unnormalized
                     log-posterior of parameter params_i.
        """
        llik = self.ll(u_params, data)
        lprior = self.prior.log_prob(self.transform.to(u_params))
        return llik + lprior

    @abstractmethod
    def simulate(self, c_params:torch.Tensor, s0:torch.Tensor, T:float|torch.Tensor, M:int) -> torch.Tensor:
        """
        Simulate M paths of asset prices. 

        Args:
            c_params (torch.Tensor): Tensor of shape (N, D), containing N sets of
                                    constrained parameters.
            s0 (torch.Tensor): Tensor of shape (1,), the initial value of each path.
            T (float or torch.Tensor):  Total simulation time.
            M (int): Number of paths to simulate.

        Returns:
            torch.Tensor: Tensor of shape (L, M), where L is the number of time steps
                      determined by T and self.dt.
        """
        pass
