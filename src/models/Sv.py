import torch
import numpy as np
from src.utils.priors import IndependentPrior
from src.utils.optimization import IndependentTransform
from src.utils.special_functions import logit, inv_logit

from PyMB import PyMB_model, Laplace

class Sv():
    def __init__(self, dt: list[float, torch.Tensor], prior: IndependentPrior):
        if isinstance(dt, float):
            self.dt = torch.tensor(dt)
        else:
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

        self.m.data['dt'] = self.dt.item()
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
        self.m_vec.data['dt'] = self.dt.item()

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
    
    def inv_transform_tmb(self, tmb_params):
        mu, log_sigma_y, log_sigma_h, logit_phi, logit_rho = tmb_params.T
        c_params = torch.stack([
            mu, log_sigma_y.exp(), log_sigma_h.exp(),
            inv_logit(logit_phi), inv_logit(logit_rho)
        ], dim=1)
        return c_params

    def get_latent(self, with_std=False):
        report = torch.tensor(self.m.get_random_report(), dtype=torch.float32)
        #report = torch.tensor(self.m.get_report())
        h = report[:, 0]
        std = report[:, 1]
        if with_std:
            return h, std
        return h

    def get_cov_fixed(self, map_opt):
        map_nat = self.transform.to(map_opt)
        map_tmb = self.transform_tmb(map_nat)

        print(map_nat)
        cov_np = self.m.get_cov_fixed(map_tmb.numpy())  # Cov of natural params
        cov = torch.tensor(cov_np, dtype=torch.float32)

        eigenvals = torch.linalg.eigvalsh(cov)
        print(eigenvals)
        return cov

    def sum_ll(self, opt_params, data):
        return NotImplementedError
        # Expects params in optimization parametrization
        natural_params = self.transform.to(opt_params)
        tmb_params = self.transform_tmb(natural_params)

        model = self.m_vec.TMB.model
        model = dict(zip(model.names(), model))
        sum_nll = Laplace.apply(tmb_params, model)
        return -sum_nll

    def ll(self, opt_params, data):
        # Expects params in optimization parametrization
        self.build_objective(data)
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
    
    def compute_map(self, u_init, data, lr, n_steps, verbose=False):
        u_params = u_init.detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(params=[u_params], lr=lr, max_iter=n_steps)
        closure_calls = 0
        def closure():
            nonlocal closure_calls
            closure_calls += 1
            optimizer.zero_grad()
            loss = -self.lpost(u_params, data)
            if verbose: print(f'Loss: {loss.item():.3f}')
            loss.backward()
            return loss
        optimizer.step(closure)
        if closure_calls >= n_steps:
            raise ValueError("MAP optimization didn't converge.")

        u_map = u_params.detach()
        return u_map

    def compute_cov(self, map_c):
        map_tmb = self.transform_tmb(map_c)
        nll_hessian_np = self.m.get_hessian(map_tmb.numpy())
        nll_hessian = torch.tensor(nll_hessian_np, dtype=torch.float32).squeeze()

        nlprior = lambda tmb_params: -self.prior.log_prob(self.inv_transform_tmb(tmb_params))
        nlprior_hessian = torch.func.hessian(nlprior, argnums=0)(map_tmb).squeeze()

        hessian = nll_hessian + nlprior_hessian  # Log posterior Hessian in the TMB parametrization
        cov_tmb = torch.linalg.inv(hessian)
        return cov_tmb

    def sample_posterior(self, map_c, n_samples):
        cov_tmb = self.compute_cov(map_c)
        map_tmb = self.transform_tmb(map_c)
        posterior = torch.distributions.MultivariateNormal(
            loc=map_tmb.squeeze(),
            covariance_matrix=cov_tmb
        )
        samples_tmb = posterior.sample(sample_shape=(n_samples,))
        samples_c = self.inv_transform_tmb(samples_tmb)
        samples_u = self.transform.inv(samples_c)
        return samples_u

    def params_uncertainty(self, u_init, data, n_particles,
                           lr, n_steps, verbose=False):
        self.build_objective(data)

        u_map = self.compute_map(u_init, data, lr, n_steps, verbose)
        c_map = self.transform.to(u_map)
        u_particles = self.sample_posterior(c_map, n_particles)
        return u_map, u_particles

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
