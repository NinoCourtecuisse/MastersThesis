import torch
from utils.NormalInverseGaussian import NormalInverseGaussian
from utils.InverseGaussian import InverseGaussian

class Nig():
    def __init__(self, dt: float):
        self.dt = dt

    def reparam(self, params):
        # Reparametrize: natural -> optimization
        mu = params[:, 0]
        log_sigma = torch.log(params[:, 1])
        xi = params[:, 2]
        log_eta = torch.log(params[:, 3])

        new_params = torch.stack([mu, log_sigma, xi, log_eta], dim = 1)
        return new_params
    
    def inverse_reparam(self, params):
        # Reparametrize: optimization -> natural
        mu = params[:, 0]
        sigma = torch.exp(params[:, 1])
        xi = params[:, 2]
        eta = torch.exp(params[:, 3])

        new_params = torch.stack([mu, sigma, xi, eta], dim = 1)
        return new_params
    
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

    def log_transition(self, params, s, s_next):
        # Expects params in optimization parametrization
        natural = self.inverse_reparam(params)
        original = self.reparam_nat_orig(natural)
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
        return torch.relu(log_transition + 15) - 15

    def ll(self, params: torch.tensor, data: torch.tensor):
        # Expects params in optimization parametrization
        s = data[:-1]
        s_next = data[1:]
        log_transitions = self.log_transition(params, s, s_next)
        ll = torch.sum(log_transitions, dim = 1)
        return ll

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
