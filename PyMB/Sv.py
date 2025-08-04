import torch
import numpy as np
from src.utils.priors import IndependentPrior
from src.utils.optimization import IndependentTransform
from src.utils.special_functions import logit

from PyMB import PyMB_model, Laplace

class Sv():
    def __init__(self, dt: float|torch.Tensor, prior: IndependentPrior):
        if isinstance(dt, float):
            self.dt = torch.tensor(dt)
        else:
            self.dt = dt
        self.prior = prior
        self.transform = IndependentTransform(prior)

        # Instantiate the TMB Object from the C++ implementation of the joint-likelihood
        self.m = PyMB_model(name='sv')
        self.m.load_model(so_file='PyMB/likelihoods/tmb_tmp/sv.so')

    def build_objective(self, data:torch.Tensor):
        """
        TMB requires to first build an "objective function" by specifying
        (i) Initial parameters
        (ii) Data on which to evaluate the likelihood
        (iii) Which variable we want to marginalize out of the joint likelihood

        Note: the Initial parameters will not be used since we do not use the built-in
        optimization of TMB, but instead use PyTorch optimizer. TMB is only used to evaluate
        the Laplace approximation and its gradients.

        Args:
            data (torch.Tensor): Tensor of shape (T+1,) containing asset prices
                                {s_0, ..., s_T}. 
        """
        self.m.init['mu'] = 0.0     # Dummy, will not be used
        self.m.init['log_sigma_y'] = 0.0
        self.m.init['log_sigma_h'] = 0.0
        self.m.init['logit_phi'] = 3.0
        self.m.init['logit_rho'] = 0.0

        self.m.data['dt'] = self.dt.item()
        log_returns = torch.log(data[1:] / data[:-1])
        self.m.data['y'] = log_returns.numpy()
        self.m.init['h'] = torch.zeros_like(log_returns).numpy()    # Log-variance

        self.m.build_objective_function(random=['h'], silent=True)  # Specify the log-variance as latent variable

    def transform_tmb(self, c_params:torch.Tensor) -> torch.Tensor:
        """
        Map the constrained params to the parametrization used by TMB.

        Args:
            c_params (torch.Tensor): Tensor of shape (N, D), containing N sets of
                                    constrained parameters.
        Returns:
            torch.Tensor: Tensor of shape (N, D), , containing N sets of
                            TMB parameters.
        """
        mu, sigma_y, sigma_h, phi, rho = c_params.T
        log_sigma_y = torch.log(sigma_y)
        log_sigma_h = torch.log(sigma_h)
        logit_phi = logit(phi)
        logit_rho = logit(rho)

        tmb_params = torch.stack([mu, log_sigma_y, log_sigma_h, logit_phi, logit_rho], dim=1)
        return tmb_params

    def get_latent(self, with_std:bool=False):
        """
        Get the latent variable (=log-variance). It will return the estimated path
        that was the most likely among all parameters given as input so far.

        Args:
            with_std (bool): Flag to return or not the standard deviation of the latent variable,
                estimated from the Hessian.
        """
        report = torch.tensor(self.m.get_random_report(), dtype=torch.float32)
        h = report[:, 0]
        std = report[:, 1]
        if with_std:
            return h, std
        return h

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
        # Expects params in optimization parametrization
        self.build_objective(data)
        c_params = self.transform.to(u_params)
        tmb_params = self.transform_tmb(c_params)

        model = self.m.TMB.model
        model = dict(zip(model.names(), model))
        nll = []                # Negative log-likelihood (Laplace approximation)
        for i in range(tmb_params.shape[0]):
            la = Laplace.apply(tmb_params[i, :], model)
            nll.append(la)
        nll = torch.stack(nll, dim=0).squeeze()
        return -nll

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

    def simulate(self, c_params, s0, T, M):
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
        with torch.no_grad():
            mu, sigma_y, sigma_h, phi, rho = c_params.T
            N = int(torch.round(T / self.dt).item())

            returns = torch.zeros(size=(N - 1, M))  # log-returns
            h = torch.zeros(size=(N, M))            # log-variance

            h[0, :] = (sigma_h * np.sqrt(self.dt) / torch.sqrt(1 - phi**2)) * torch.randn(size=(1, M))
            for i in range(1, N):
                h[i, :] = phi * h[i - 1, :] + sigma_h * np.sqrt(self.dt) * torch.randn(size=(1, M))
                eta = (h[i, :] - phi * h[i - 1, :]) / (sigma_h * np.sqrt(self.dt))
                returns[i - 1, :] = mu * self.dt + sigma_y * np.sqrt(self.dt) * torch.exp(h[i, :] / 2) \
                            * (rho * eta + torch.sqrt(1 - rho**2) * torch.randn(size=(1, M)))

            x = torch.cumsum(returns, dim=0) + torch.log(s0)
            x = torch.cat([torch.log(s0) * torch.ones((1, M)), x], dim = 0)
            s = torch.exp(x)
        return s, h
