import torch
from torch import distributions as D
from utils.distributions import InverseGaussian, NormalInverseGaussian
from utils.priors import NigPrior, IndependentPrior
from utils.optimization import NigTransform, IndependentTransform

class Nig():
    def __init__(self, dt: list[float, torch.Tensor], prior: list[IndependentPrior, NigPrior]):
        if isinstance(dt, float):
            self.dt = torch.tensor(dt)
        else:
            self.dt = dt
        self.prior = prior
        if isinstance(prior, IndependentPrior):
            self.transform = IndependentTransform(prior)
        else:
            self.transform = NigTransform(prior)

    def log_transition(self, opt_params, s, s_next):
        # Expects params in optimization parametrization
        params = self.transform.to(opt_params)
        mu, sigma, xi, eta = params.T.unsqueeze(2)  # shape (4, N, 1)
        dt = self.dt

        log_return = torch.log(s_next / s)
        log_transition = NormalInverseGaussian.from_moments(
            mu=dt * mu,
            sigma=torch.sqrt(dt) * sigma,
            xi=torch.sqrt(dt) * xi,
            eta=eta / dt
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

    def compute_map(self, u_init, data, lr, n_steps, verbose=False):
        u_params = u_init.detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(params=[u_params], lr=lr, max_iter=n_steps)
        closure_calls = 0
        def closure():
            nonlocal closure_calls
            closure_calls += 1
            optimizer.zero_grad()
            loss = -self.lpost(u_params, data)     # Minimize nll
            if verbose: print(f'Loss: {loss.item():.3f}')
            loss.backward()
            return loss
        optimizer.step(closure)
        if closure_calls >= n_steps:
            raise ValueError("MAP optimization didn't converge.")

        u_map = u_params.detach()
        return u_map

    def params_uncertainty(self, u_init, data, n_particles,
                           lr, n_steps, verbose=False):
        u_map = self.compute_map(u_init, data, lr, n_steps, verbose)

        nll = lambda u_params: -self.lpost(u_params, data)
        hessian = torch.autograd.functional.hessian(nll, u_map).detach().squeeze()
        u_cov = torch.linalg.inv(hessian)

        d = D.MultivariateNormal(
            loc=u_map.squeeze(),
            covariance_matrix=u_cov
        )
        u_particles = d.sample(sample_shape=(n_particles,))
        return u_map, u_particles

    def simulate(self, params, s0, T, M):
        # Expects params in natural parametrization
        with torch.no_grad():
            mu, sigma, xi, eta = params.T.unsqueeze(2)
            mu_, alpha, beta, delta = NormalInverseGaussian.reparametrize(mu, sigma, xi, eta)
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
