import torch
from torch.distributions import Normal, Poisson, Chi2
import math

class Heston(torch.nn.Module):
    def __init__(self, mu, k, theta, sigma, rho, v0):
        super().__init__()
        """
        Reparametrize to enforce the following constraints:
        0 < k ; 0 < theta ; 0 < sigma ; -1 < rho < 1
        """
        #self.mu = torch.nn.Parameter(mu)
        self.mu = mu
        self.l_k = torch.nn.Parameter(torch.log(k))             # k -> log(k)
        self.l_theta = torch.nn.Parameter(torch.log(theta))     # theta -> log(theta)
        self.l_sigma = torch.nn.Parameter(torch.log(sigma))     # sigma -> log(sigma)
        self.at_rho = torch.nn.Parameter(torch.atanh(rho))      # rho -> atanh(rho)

        self.v0 = v0
        self.params_names = ['mu', 'k', 'theta', 'sigma', 'rho', 'v0']
        self.model_type = 'SV'

    def inv_reparam(self):
        # Inverse the reparametrization.
        return self.mu, torch.exp(self.l_k), torch.exp(self.l_theta), torch.exp(self.l_sigma), torch.tanh(self.at_rho), self.v0
    
    def step(self, x0, v0, delta_t):
        """
        Sample (x_dt, v_dt) from (x_0, v_0) using NumPy to access the ncx2 distribution.
        """
        # Exact sampling for variance
        mu, k, theta, sigma, rho, _ = self.inv_reparam()
        v_next = self.sample_variance(1, v0, delta_t)

        # Simplified Broadie-Kaya for log-price
        mean = x0 + mu * delta_t + (k * rho / sigma - 0.5) * v0 * delta_t + rho * (v_next - v0 - k * theta * delta_t) / sigma
        var = (1 - rho**2) * v0 * delta_t
        x_next = Normal(loc=mean, scale=torch.sqrt(var)).sample(sample_shape=torch.Size((1,)))
        return x_next, v_next
    
    def simulate(self, x0, v0, delta_t, N):
        """
        Produce one path of log-price process and variance process.
        """
        with torch.no_grad():
            log_spot_path = torch.zeros(size=(N,))
            variance_path = torch.zeros(size=(N,))

            log_spot_path[0] = x0
            variance_path[0] = v0
            for i in range(1, N):
                X, V = self.step(log_spot_path[i-1], variance_path[i-1], delta_t)
                log_spot_path[i] = X
                variance_path[i] = V
        return log_spot_path, variance_path
    
    def update_v0(self, v0):
        self.v0 = v0

    def euler_transition(self, s, v, s_next, delta_t=1/252):
        """
        Evaluates p_(log(s_next) ; log(s), v) over a time-step delta_t. 
        Approximated by a Normal distribution (Euler discretization).
        """
        transition = torch.zeros_like(s_next)
        mu, k, theta, sigma, rho, v0 = self.inv_reparam()

        mean = torch.log(s) + (mu - 0.5 * v) * delta_t
        var = v * delta_t
        transition = torch.exp(Normal(loc=mean, scale=torch.sqrt(var)).log_prob(torch.log(s_next)))
        return torch.relu(transition - 1e-6) + 1e-6
    
    def sample_variance(self, n_samples, v, delta_t):
        # Sample the next variance from ncx2
        mu, k, theta, sigma, rho, _ = self.inv_reparam()
        n = 4 * k * torch.exp(-k * delta_t) / (sigma**2 * (1 - torch.exp(-k * delta_t)))
        d = 4 * k * theta / sigma**2
        N = Poisson(rate = v * n / 2).sample(sample_shape=torch.Size((n_samples,)))

        if v.dim() == 0:
            N = N.unsqueeze(1)
        v1 = torch.exp(-k * delta_t) / n * Chi2(df = d + 2 * N).sample()
        return v1
    
    def mc_transition(self, s, v, s_next, delta_t=1/252):
        """
        Evaluates p_(log(s_next) ; log(s), v) over a time-step delta_t.
        """
        transition = torch.zeros_like(s_next)
        mu, k, theta, sigma, rho, _ = self.inv_reparam()
        
        n_samples = 10000
        v_next = self.sample_variance(n_samples, v, delta_t)    # Shape: (n_samples, len(v))

        # Simplified Broadie-Kaya
        mean = torch.log(s).unsqueeze(0) + mu * delta_t + (k * rho / sigma - 0.5) * v_next * delta_t + rho * (v_next - v.unsqueeze(0) - k * theta * delta_t) / sigma     # (n_mc, len(s0))   
        var = (1 - rho**2) * v * delta_t   # (len(v),)
        transition = torch.mean(torch.exp(Normal(loc=mean, scale=torch.sqrt(var)).log_prob(torch.log(s_next))), dim=0)  # (len(s1),)
        return torch.relu(transition - 1e-6) + 1e-6
    
    def variance_path_vec(self, spot_prices, delta_t):
        # Generate a deterministic path, conditional on the spot path
        mu, k, theta, sigma, rho, v0 = self.inv_reparam()
        log_spot_diff = torch.log(spot_prices[1:]) - torch.log(spot_prices[:-1])
        var_path = torch.zeros_like(spot_prices)
        var_path[0] = v0
        beta = k * theta * delta_t + rho * sigma * log_spot_diff - rho * sigma * mu * delta_t
        alpha = 1 - k * delta_t + 0.5 * rho * sigma * delta_t

        chunk_size = 64
        exp = torch.arange(chunk_size).unsqueeze(1) - torch.arange(chunk_size).unsqueeze(0)
        exp = torch.tril(exp)
        M = torch.tril(alpha ** exp)
        alpha_powers = alpha ** (torch.arange(1, chunk_size+1))
        t = 1
        T = len(spot_prices)
        n_iter = math.ceil((T - 1) / chunk_size)
        v_prev = v0
        for i in range(n_iter):
            t = i * chunk_size + 1
            end = min(t + chunk_size, T)
            chunk_len = end - t
            chunk_powers = alpha_powers[:chunk_len]
            M_chunk = M[:chunk_len, :chunk_len]
            beta_chunk = beta[t-1:end-1]
            prod_1 = chunk_powers * v_prev
            var_chunk = prod_1 + M_chunk @ beta_chunk
            var_path[t:t + chunk_len] = var_chunk
            v_prev = var_chunk[-1].clone()
        return torch.relu(var_path - 1e-4) + 1e-4
    
    def variance_path(self, spot_prices, delta_t):
        # Generate a deterministic path, conditional on the spot path
        mu, k, theta, sigma, rho, v0 = self.inv_reparam()
        log_spot_diff = torch.log(spot_prices[1:]) - torch.log(spot_prices[:-1])
        var_path = torch.zeros_like(spot_prices)
        beta = k * theta * delta_t + rho * sigma * log_spot_diff - rho * sigma * mu * delta_t
        alpha = 1 - k * delta_t + 0.5 * rho * sigma * delta_t
        var_path[0] = v0
        for i in range(1, len(var_path)):
            #dX = torch.log(spot_prices[i]) - torch.log(spot_prices[i-1])
            #v_prev = var_path[i-1]
            #beta_prev = beta[i-1]
            var_path[i] = beta[i-1] + alpha * var_path[i-1]
            #var_path[i] = v_prev + k * (theta - v_prev) * delta_t + rho * sigma * (dX - (mu - 0.5 * v_prev) * delta_t) #+ sigma * torch.sqrt(1 - rho**2) * v_prev * delta_t * torch.randn(1)
        return torch.relu(var_path - 1e-4) + 1e-4
    
    def variance_path_mm(self, spot_prices, delta_t):
        mu, k, theta, sigma, rho, v0 = self.inv_reparam()
        log_spot_diff = torch.log(spot_prices[1:]) - torch.log(spot_prices[:-1])
        var_path = torch.zeros_like(spot_prices)
        var_path[0] = v0
        T = len(spot_prices)
        for t in range(1, T):
            v_prev = torch.relu(var_path[t-1].clone() - 1e-2) + 1e-2
            m = theta + (v_prev - theta) * torch.exp(-k * delta_t)
            s2 = v_prev * sigma**2 * torch.exp(-k * delta_t) * (1 - torch.exp(-k * delta_t)) / k + theta * sigma**2 * (1 - torch.exp(-k * delta_t))**2 / (2 * k)
            sigma_tilde = torch.sqrt(torch.log(1 + s2 / m**2))
            raw_var = m * torch.exp( rho * sigma_tilde / torch.sqrt(delta_t * v_prev) * (log_spot_diff[t-1] - (mu - 0.5 * v_prev) * delta_t) - 0.5 * rho**2 * sigma_tilde**2 )
            if raw_var > 2.0:
                raw_var = var_path[t-1]
            var_path[t] = raw_var
        return var_path

    def forward(self, spot_prices, t, window=100, delta_t=1/252, update_v0=False):
        # Generate a variance path
        start = max(t - window, 0)
        self.mu = torch.mean((spot_prices[start+1:t+1] - spot_prices[start:t]) / spot_prices[start:t])
        var_path = self.variance_path_mm(spot_prices[start:t], delta_t)
        
        # Evaluate transition densities
        transition_evals = self.euler_transition(spot_prices[start:t], var_path, spot_prices[start+1:t+1], delta_t)
        log_likelihood = transition_evals.log().sum()
        if update_v0:
            self.v0 = var_path[1].detach().clone()
        #decay = neta**(torch.arange(t - 1, -1, -1))
        #log_likelihood = decay.inner(transition_evals.log())
        return log_likelihood
    
    def _heston_integrand(self, s, v, s_next, u, delta_t):
        """
        Integrand for the FT on the characterstic function (CF).
        Compute the CF: phi(u) = E[exp(i*u * log(s_next)) ; log(s), v], i.e. the CF of log(s_next)
        Compute the integrand: y(u) = Re(exp(-i * log(s_next) * u) * phi(u)), i.e. the integrand for the IFT
        """
        mu, k, theta, sigma, rho, v0 = self.inv_reparam()
        x = torch.log(s)
        a = k * theta
        b = k
        tmp = - 0.5
        d = torch.sqrt((rho * sigma * 1j * u - b)**2 - sigma**2 * (2 * tmp * 1j * u - u**2))
        g = (b - rho * sigma * 1j * u + d) / (b - rho * sigma * 1j * u - d)

        c = 1 / g
        D = (b - rho * sigma * 1j * u - d) * (1 - torch.exp(-d * delta_t)) / (sigma**2 * (1 - c * torch.exp(-d * delta_t)))
        C = mu * 1j * u * delta_t + a * ( (b - rho * sigma * 1j * u - d) * delta_t - 2 * torch.log( (1 - c * torch.exp(- d * delta_t)) / (1 - c) ) ) / sigma**2
        f = torch.exp(C.unsqueeze(1) + D.unsqueeze(1) * v.repeat(len(u),1) + 1j * x.repeat(len(u), 1) * u.unsqueeze(1))
        y = torch.real( torch.exp(-1j * torch.log(s_next).repeat(len(u),1) * u.unsqueeze(1)) * f )
        return y

    def ft_transition(self, s, v, s_next, delta_t):
        """
        Exact transition density of the loc-price process, via IFT on the CF.
        Compute p(log(s_next) ; log(s), v) = integral(y(u) du), using the trapezoid rule.
        """
        relu = torch.nn.ReLU()
        u = torch.linspace(1e-4, 400, 500)
        y = self._heston_integrand(s, v, s_next, u, delta_t)
        pdf = torch.trapezoid(y, u, dim=0) / torch.tensor(math.pi)
        return relu(pdf - 1e-6) + 1e-6
