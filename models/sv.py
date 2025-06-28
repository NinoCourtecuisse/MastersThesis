import torch
import numpy as np
from utils.priors import IndependentPrior
from utils.optimization import IndependentTransform
from utils.special_functions import logit

from PyMB import PyMB_model, Laplace

class Sv():
    def __init__(self, dt: float, prior: IndependentPrior):
        self.dt = dt
        self.prior = prior
        self.transform = IndependentTransform(prior)

        self.m_vec = PyMB_model(name='sv_vec')
        self.m_vec.load_model(so_file='PyMB/likelihoods/tmb_tmp/sv_vec.so')
        self.m = PyMB_model(name='sv')
        self.m.load_model(so_file='PyMB/likelihoods/tmb_tmp/sv.so')
    
    def build_objective(self, data):
        self.m.init['mu'] = 0.0
        self.m.init['log_sigma_y'] = 0.0  # Dummy parameters, will not be used
        self.m.init['log_sigma_h'] = 0.0
        self.m.init['logit_phi'] = 3.0
        self.m.init['logit_rho'] = 0.0

        self.m.data['dt'] = self.dt
        log_returns = torch.log(data[1:] / data[:-1])
        self.m.data['y'] = log_returns.numpy()
        self.m.init['h'] = torch.zeros_like(log_returns).numpy()
        self.m.build_objective_function(random=['h'], silent=True)
    
    def build_objective_vec(self, data, n_particles):
        self.m_vec.init['mu'] = np.zeros((n_particles,))        # Dummy parameters, will not be used
        self.m_vec.init['log_sigma_y'] = np.zeros((n_particles,))
        self.m_vec.init['log_sigma_h'] = np.zeros((n_particles,))
        self.m_vec.init['logit_phi'] = 3 * np.ones((n_particles,))
        self.m_vec.init['logit_rho'] = np.zeros((n_particles,))

        log_returns = torch.log(data[1:] / data[:-1])
        self.m_vec.data['N'] = len(log_returns)
        self.m_vec.data['P'] = n_particles
        self.m.data['dt'] = self.dt

        rep_ret = np.tile(log_returns.numpy()[:, None], (1, n_particles)).T
        self.m_vec.data['y'] = rep_ret
        self.m_vec.init['h'] = np.zeros_like(rep_ret)
        self.m_vec.build_objective_function(random=['h'], silent=True)

    def transform_tmb(self, natural_params):
        # Natural params: mu, sigma_y, sigma_h, phi, rho (annualized)
        mu = natural_params[:, 0]
        log_sigma_y = torch.log(natural_params[:, 1])
        log_sigma_h = torch.log(natural_params[:, 2])
        logit_phi = logit(natural_params[:, 3])
        logit_rho = logit(natural_params[:, 4])

        tmb_params = torch.stack([mu, log_sigma_y, log_sigma_h, logit_phi, logit_rho], dim=1)
        return tmb_params

    def get_latent(self, with_std=False):
        report = torch.tensor(self.m.get_report())
        h = report[:, 0]
        std = report[:, 1]
        if with_std:
            return h, std
        return h

    def sum_ll(self, opt_params, data):
        # Expects params in optimization parametrization
        natural_params = self.transform.to(opt_params)
        tmb_params = self.transform_tmb(natural_params)

        model = self.m_vec.TMB.model
        model = dict(zip(model.names(), model))
        sum_nll = Laplace.apply(tmb_params, model)
        return -sum_nll

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

    def lpost(self, opt_params, data):
        llik = self.ll(opt_params, data)
        lprior = self.prior.log_prob(self.transform.to(opt_params))
        return llik + lprior

    def simulate(self, params, s0, T, M):
        with torch.no_grad():
            mu = params[:, 0]
            sigma_y = params[:, 1]
            sigma_h = params[:, 2]
            phi = params[:, 3]
            rho = params[:, 4]
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
        return x, h
