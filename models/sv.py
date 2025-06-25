import torch
from utils.priors import IndependentPrior
from utils.optimization import IndependentTransform
from utils.special_functions import logit

from PyMB import PyMB_model, Laplace

class Sv():
    def __init__(self, dt: float, prior: IndependentPrior):
        self.dt = dt
        self.prior = prior
        self.transform = IndependentTransform(prior)

        self.m = PyMB_model(name='sv')
        self.m.load_model(so_file='PyMB/likelihoods/tmb_tmp/sv.so')
        self.m.init['log_sigma_y'] = 0  # Dummy parameters, will not be used
        self.m.init['log_sigma_h'] = 0
        self.m.init['logit_phi'] = 3
        self.m.init['logit_rho'] = 0
        self.m.init['mu'] = 0
    
    def build_objective(self, data):
        self.m.data['y'] = data.numpy()
        self.m.init['h'] = torch.zeros_like(data).numpy()
        self.m.build_objective_function(random=['h'], silent=True)
    
    def transform_tmb(self, natural_params):
        log_sigma_y = natural_params[:, 0].log()
        log_sigma_h = natural_params[:, 1].log()
        logit_phi = logit(natural_params[:, 2])
        logit_rho = logit(natural_params[:, 3])
        mu = natural_params[:, 4]

        tmb_params = torch.stack([log_sigma_y, log_sigma_h, logit_phi, logit_rho, mu], dim=1)
        return tmb_params

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
