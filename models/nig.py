import torch
from utils.distributions import InverseGaussian, NormalInverseGaussian
from utils.priors import IndependentPrior
from utils.optimization import IndependentTransform, NigTransform

class Nig():
    def __init__(self, dt: list[float, torch.Tensor], prior: IndependentPrior):
        if isinstance(dt, float):
            self.dt = torch.tensor(dt)
        else:
            self.dt = dt
        self.prior = prior
        self.transform = NigTransform(prior)
        self.is_sv = False

    def log_transition(self, opt_params, s, s_next):
        # Expects params in optimization parametrization
        params = self.transform.to(opt_params)
        mu, sigma, gamma_1, gamma_2 = params.T.unsqueeze(2)  # shape (4, N, 1)
        dt = self.dt

        log_return = torch.log(s_next / s)
        log_transition = NormalInverseGaussian.from_moments(
            mu = dt * mu,
            sigma = torch.sqrt(dt) * sigma,
            gamma_1 = gamma_1 / torch.sqrt(dt),
            gamma_2 = gamma_2 / dt
        ).log_prob(log_return)
        return log_transition

    def ll(self, opt_params, data):
        # Expects params in optimization parametrization
        s = data[:-1]
        s_next = data[1:]
        log_transitions = self.log_transition(opt_params, s, s_next)
        ll = torch.sum(log_transitions, dim = 1)
        return ll

    def lpost(self, opt_params, data):
        llik = self.ll(opt_params, data)
        lprior = self.prior.log_prob(self.transform.to(opt_params))
        return llik + lprior

    def simulate(self, params, s0, T, M):
        # Expects params in natural parametrization
        with torch.no_grad():
            mu, sigma, gamma_1, gamma_2 = params.T.unsqueeze(2)
            mu_, alpha, beta, delta = NormalInverseGaussian.reparametrize(mu, sigma, gamma_1, gamma_2)
            dt = self.dt
            n = int(torch.round(T / dt).item())

            # Generate M paths from IG process
            dI = InverseGaussian(
                mu = delta * dt / torch.sqrt(alpha**2 - beta**2),
                lam = (delta * dt)**2
            ).sample(sample_shape=(n - 1, M))
            I = torch.cumsum(dI, dim=0)

            # Generate M paths from NIG process
            Z = torch.randn(size=(n - 1, M))
            W_It = torch.cumsum(torch.sqrt(dI) * Z, dim=0)
            X = mu_ * torch.linspace(dt, T, n - 1).unsqueeze(1) + beta * I + W_It

            s = s0 * torch.ones((n, M))
            s[1:, :] = s0 * torch.exp(X)
        return s
