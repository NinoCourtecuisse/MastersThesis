import torch
from utils.distributions import InverseGaussian, NormalInverseGaussian
from utils.priors import IndependentPrior
from utils.optimization import IndependentTransform

class Nig():
    def __init__(self, dt: float, prior: IndependentPrior):
        self.dt = dt
        self.prior = prior
        self.transform = IndependentTransform(prior)

    def reparam_nat_orig(self, params):
        # Reparametrize: natural -> original
        mu = params[:, 0]
        sigma = params[:, 1]
        xi = params[:, 2]
        eta = params[:, 3]

        sigma_tilde = sigma / torch.sqrt(1 + eta * xi**2)

        mu_tilde = -sigma_tilde * xi + mu
        alpha_tilde = torch.sqrt(1/eta + xi**2) / sigma_tilde
        beta_tilde = xi / sigma_tilde
        delta_tilde = sigma_tilde / torch.sqrt(eta)
        original = torch.stack([mu_tilde, alpha_tilde, beta_tilde, delta_tilde], dim = 1)
        return original

    def log_transition(self, opt_params, s, s_next):
        # Expects params in optimization parametrization
        params = self.transform.to(opt_params)
        original = self.reparam_nat_orig(params)
        mu = original[:, 0].unsqueeze(1)
        alpha = original[:, 1].unsqueeze(1)
        beta = original[:, 2].unsqueeze(1)
        delta = original[:, 3].unsqueeze(1)
        dt = self.dt

        log_return = torch.log(s_next / s)
        log_transition = NormalInverseGaussian(
            mu=dt * mu,
            alpha=alpha,
            beta=beta,
            delta=delta * dt
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
            original = self.reparam_nat_orig(params)
            dt = self.dt
            mu = original[:, 0].unsqueeze(1)
            alpha = original[:, 1].unsqueeze(1)
            beta = original[:, 2].unsqueeze(1)
            delta = original[:, 3].unsqueeze(1)

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
            X = mu * torch.linspace(dt, T, n - 1).unsqueeze(1) + beta * I + W_It

            s = s0 * torch.ones((n, M))
            s[1:, :] = s0 * torch.exp(X)
        return s
