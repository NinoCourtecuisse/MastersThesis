import torch
from src.utils.priors import IndependentPrior
from src.utils.optimization import IndependentTransform
from src.utils.special_functions import logit

from PyMB import PyMB_model, Laplace

class Sabr():
    def __init__(self, dt: float|torch.Tensor, prior: IndependentPrior):
        if isinstance(dt, float):
            self.dt = torch.tensor(dt)
        else:
            self.dt = dt
        self.prior = prior
        self.transform = IndependentTransform(prior)

        # Instantiate the TMB Object from the C++ implementation of the joint-likelihood
        self.m = PyMB_model(name='sabr')
        self.m.load_model(so_file='PyMB/likelihoods/tmb_tmp/sabr.so')

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
        self.m.init['log_beta'] = 0.0
        self.m.init['log_sigma'] = 0.0
        self.m.init['logit_rho'] = 0.0

        self.m.data['dt'] = self.dt.item()
        self.m.data['X'] = data.log().numpy()
        self.m.init['h'] = torch.zeros_like(data).numpy()               # Log-variance
        self.m.build_objective_function(random=['h'], silent=True)      # Specify the log-variance as latent variable

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
        mu, beta, sigma, rho = c_params.T
        log_beta = torch.log(beta)
        log_sigma = torch.log(sigma)
        logit_rho = logit(rho)

        tmb_params = torch.stack([mu, log_beta, log_sigma, logit_rho], dim=1)
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

    def local_var(self, s:torch.Tensor, log_delta:torch.Tensor, beta:torch.Tensor) -> torch.Tensor:
        var = torch.exp(2 * log_delta) * s**(beta - 2)
        return var

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
