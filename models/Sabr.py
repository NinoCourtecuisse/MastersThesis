import torch
import numpy as np
from utils.priors import IndependentPrior
from utils.optimization import IndependentTransform
from utils.special_functions import logit

from PyMB import PyMB_model, Laplace

class Sabr():
    def __init__(self, dt: float, prior: IndependentPrior):
        self.dt = dt
        self.prior = prior
        self.transform = IndependentTransform(prior)

        self.m = PyMB_model(name='sabr')
        self.m.load_model(so_file='PyMB/likelihoods/tmb_tmp/sabr.so')

    def build_objective(self, data):
        self.m.init['mu'] = 0.0
        self.m.init['log_beta'] = 0.0
        self.m.init['log_sigma'] = 0.0
        self.m.init['logit_rho'] = 0.0

        self.m.data['dt'] = self.dt
        self.m.data['X'] = data.log().numpy()
        self.m.init['h'] = torch.zeros_like(data).numpy()
        self.m.build_objective_function(random=['h'], silent=True)
    
    def transform_tmb(self, natural_params):
        # Natural params: mu, beta, sigma, rho (annualized)
        mu = natural_params[:, 0]
        log_beta = natural_params[:, 1].log()
        log_sigma = natural_params[:, 2].log()
        logit_rho = logit(natural_params[:, 3])

        tmb_params = torch.stack([mu, log_beta, log_sigma, logit_rho], dim=1)
        return tmb_params

    def get_latent(self, with_std=False):
        report = torch.tensor(self.m.get_report())
        h = report[:, 0]
        std = report[:, 1]
        if with_std:
            return h, std
        return h

    def local_var(self, s, log_delta, beta):
        var = torch.exp(2 * log_delta) * s**(beta - 2)
        return var

    def ll(self, opt_params, data):
        # Expects params in optimization parametrization
        natural_params = self.transform.to(opt_params)
        tmb_params = self.transform_tmb(natural_params)

        model = self.m.TMB.model
        model = dict(zip(model.names(), model))
        nll = []
        for i in range(tmb_params.shape[0]):
            la = Laplace.apply(tmb_params[i, :], model)
            nll.append(la)
        nll = torch.stack(nll, dim=0).squeeze()
        return -nll