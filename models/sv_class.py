from models.model_class import Model

import torch
import numpy as np

from PyMB.PyMB.model import model as PyMB_model
from PyMB.laplace import Laplace

class Sv(Model):
    def __init__(self, sigma_y, sigma_h, phi, rho, mu, log_returns):
        params = torch.nn.ParameterDict({
            'log_sigma_y': torch.nn.Parameter(torch.log(sigma_y)),
            'log_sigma_h': torch.nn.Parameter(torch.log(sigma_h)),
            'logit_phi': torch.nn.Parameter(torch.log((1 + phi) / (1 - phi))),
            'logit_rho': torch.nn.Parameter(torch.log((1 + rho) / (1 - rho))),
            #'mu': torch.nn.Parameter(mu),
        })
        super().__init__(params)
        self.mu = mu
        self.m = PyMB_model(name='sv')
        self.m.load_model(so_file='PyMB/likelihoods/tmb_tmp/sv.so')

        self.m.data['y'] = log_returns

        self.m.init['log_sigma_y'] = 0  # Dummy parameters, will not be used
        self.m.init['log_sigma_h'] = 0
        self.m.init['logit_phi'] = 3
        self.m.init['logit_rho'] = 0
        self.m.init['mu'] = 0
        self.m.init['h'] = np.zeros_like(log_returns)

        self.m.build_objective_function(random=['h'], silent=True)
        self.model_type = 'sv'

    def inv_reparam(self):
        sigma_y = torch.exp(self.params['log_sigma_y'])
        sigma_h = torch.exp(self.params['log_sigma_h'])
        phi = (torch.exp(self.params['logit_phi']) - 1) / (torch.exp(self.params['logit_phi']) + 1)
        rho = (torch.exp(self.params['logit_rho']) - 1) / (torch.exp(self.params['logit_rho']) + 1)
        #mu = self.params['mu']
        mu = self.mu
        return sigma_y, sigma_h, phi, rho, mu
    
    def get_params(self):
        with torch.no_grad():
            sigma_y, sigma_h, phi, rho, mu = self.inv_reparam()
            params = {
                'sigma_y': sigma_y,
                'sigma_h': sigma_h,
                'phi': phi,
                'rho': rho,
                'mu': mu
            }
            return params

    def simulate(self, x0, T, dt, M):
        with torch.no_grad():
            sigma_y, sigma_h, phi, rho, mu = self.inv_reparam()
            N = int(torch.round(T / dt).item())

            returns = torch.zeros(size=(N - 1, M))
            h = torch.zeros(size=(N, M))

            h[0, :] = (sigma_h / torch.sqrt(1 - phi**2)) * torch.randn(size=(1, M))
            for i in range(1, N):
                h[i, :] = phi * h[i - 1, :] + sigma_h * torch.randn(size=(1, M))
                eta = (h[i, :] - phi * h[i - 1, :]) / sigma_h
                returns[i - 1, :] = mu + sigma_y * torch.exp(h[i, :] / 2) \
                            * (rho * eta + torch.sqrt(1 - rho**2) * torch.randn(size=(1, M)))

            x = torch.cumsum(returns, dim=0) + x0
            x = torch.cat([x0 * torch.ones((1, M)), x], dim = 0)
        return x, h

    def forward(self):
        log_sigma_y = self.params['log_sigma_y']
        log_sigma_h = self.params['log_sigma_h']
        logit_phi = self.params['logit_phi']
        logit_rho = self.params['logit_rho']
        #mu = self.params['mu']
        mu = self.mu

        model = self.m.TMB.model
        model = dict(zip(model.names(), model))

        par = torch.stack([log_sigma_y, log_sigma_h, logit_phi, logit_rho, mu])
        nll = Laplace.apply(par, model)
        return nll

    def likelihood_with_updates(self, optimizer, optimization_times, n_grad_steps, \
                                window, start, logging=None, verbose=False):
        log_returns = self.m.data['y']
        T = len(log_returns)
        log_l = torch.zeros(size=(T + 1,))

        for t in range(start, T + 1):
            self.m.data['y'] = log_returns[t - window:t]
            self.m.init['h'] = np.zeros_like(log_returns[t - window:t])
            self.m.build_objective_function(random=['h'], silent=True)

            if verbose and t % 100 == 0: print(t)

            if t in optimization_times:
                for _ in range(n_grad_steps):
                    optimizer.zero_grad()
                    loss = self.forward()
                    loss.backward()
                    optimizer.step()

            if logging is not None and t % 500 == 0:
                logging.info(f"t: {t}, " + self.print_params())

            with torch.no_grad():
                log_l[t] = - self.forward()
        return log_l