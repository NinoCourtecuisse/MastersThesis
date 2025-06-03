from models.model_class import Model

import torch
from torch.distributions import Normal, Poisson, Chi2
import math

class Heston(Model):
    def __init__(self, mu, k, theta, sigma, rho, v0):
        params = torch.nn.ParameterDict({
            'log_k': torch.nn.Parameter(torch.log(k)),
            'log_theta': torch.nn.Parameter(torch.log(theta)),
            'log_sigma': torch.nn.Parameter(torch.log(sigma)),
            'atanh_rho': torch.nn.Parameter(torch.atanh(rho))
        })
        self.mu = mu
        self.v0 = v0

        super().__init__(params)
        self.model_type = 'SV'

    def inv_reparam(self):
        # Inverse the reparametrization.
        return self.mu, torch.exp(self.params['log_k']), \
                torch.exp(self.params['log_theta']), \
                torch.exp(self.params['log_sigma']), \
                torch.tanh(self.params['atanh_rho']), self.v0

    def get_params(self):
        params = {
            'mu': self.mu.detach(),
            'k': torch.exp(self.params['log_k'].detach()),
            'theta': torch.exp(self.params['log_theta'].detach()),
            'sigma': torch.exp(self.params['log_sigma'].detach()),
            'rho': torch.tanh(self.params['atanh_rho'].detach()),
            'v0': self.v0.detach()
        }
        return params
    
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
        mu, k, theta, sigma, rho, v0 = self.inv_reparam()
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
        mu, k, theta, sigma, rho, v0 = self.inv_reparam()
        
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
    
    def variance_path_euler(self, spot_prices, delta_t):
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

    def forward(self, spot_prices, t, delta_t, window=100, update_v0=False):
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
    
    def yacine_transition(self, s, v, s_next, delta_t):
        N = 100     # Nodes
        a = 0.0
        b = 1.5
        v_next = torch.linspace(a, b, N)
        joint_evals = self.yacine_joint_transition(s, v, s_next, v_next, delta_t)
        coefs = 2.0 * torch.ones(size=(N,))
        coefs[0] = 1.0
        coefs[-1] = 1.0
        integral = joint_evals @ coefs
        integral = (b - a) * integral / (2 * N)
        return integral
    
    def yacine_joint_transition(self, s, v, s_next, v_next, delta_t):
        mu, k, theta, sigma, rho, _ = self.inv_reparam()

        a = k * theta
        Dv = 0.5 * torch.log(1 - rho**2) + torch.log(sigma) + torch.log(v_next)
        x = torch.log(s)
        x_next = torch.log(s_next)
        cm1 = (7 * sigma**4 * (x_next - x)**6) / (11520 * (-1 + rho**2)**3 * v**5) \
            - (7 * rho * sigma**3 * (x_next - x)**5 * (v_next - v)) / (1920 * (-1 + rho**2)**3 * v**5) \
            + ((-193 + 298 * rho**2) * sigma**2 * (x_next - x)**4 * (v_next - v)**2) / (11520 * (-1 + rho**2)**3 * v**5) \
            + (rho * (193 - 228 * rho**2) * sigma * (x_next - x)**3 * (v_next - v)**3) / (2880 * (-1 + rho**2)**3 * v**5) \
            + ((745 - 2648 * rho**2 + 2008 * rho**4) * (x_next - x)**2 * (v_next - v)**4) / (11520 * (-1 + rho**2)**3 * v**5) \
            + ((-745 * rho + 1876 * rho**3 - 1152 * rho**5) * (x_next - x) * (v_next - v)**5) / (5760 * (-1 + rho**2)**3 * sigma * v**5) \
            + ((945 - 2090 * rho**2 + 1152 * rho**4) * (v_next - v)**6) / (11520 * (-1 + rho**2)**3 * sigma**2 * v**5) \
            - (sigma**2 * (x_next - x)**4 * (v_next - v)) / (64 * (-1 + rho**2)**2 * v**4) \
            + (rho * sigma * (x_next - x)**3 * (v_next - v)**2) / (16 * (-1 + rho**2)**2 * v**4) \
            + ((3 - 6 * rho**2) * (x_next - x)**2 * (v_next - v)**3) / (32 * (-1 + rho**2)**2 * v**4) \
            + (rho * (-3 + 4 * rho**2) * (x_next - x) * (v_next - v)**4) / (16 * (-1 + rho**2)**2 * sigma * v**4) \
            + ((7 - 8 * rho**2) * (v_next - v)**5) / (64 * (-1 + rho**2)**2 * sigma**2 * v**4) \
            + (sigma**2 * (x_next - x)**4) / (96 * (-1 + rho**2)**2 * v**3) - (rho * sigma * (x_next - x)**3 *(v_next - v)) / (24 * (-1 + rho**2)**2 * v**3) \
            + ((-7 + 10 * rho**2) * (x_next - x)**2 * (v_next - v)**2) / (48 * (-1 + rho**2)**2 * v**3) \
            + ((7 * rho - 8 * rho**3) * (x_next - x) * (v_next - v)**3) / (24 * (-1 + rho**2)**2 * sigma * v**3) \
            + ((-15 + 16 * rho**2) * (v_next - v)**4) / (96 * (-1 + rho**2)**2 * sigma**2 * v**3) \
            - ((x_next - x)**2 * (v_next - v)) / (4 * (-1 + rho**2) * v**2) \
            + (rho * (x_next - x) * (v_next - v)**2) / (2 * (-1 + rho**2) * sigma * v**2) - (v_next - v)**3 / (4 * (-1 + rho**2) * sigma**2 * v**2) \
            + (sigma**2 * (x_next - x)**2 - 2 * rho * sigma * (x_next - x) * (v_next - v) + (v_next - v)**2) / (2 * (-1 + rho**2) * sigma**2 * v)

        c0 = (7 * sigma**4 * (x_next - x)**4) / (1920 * (-1 + rho**2)**2 * v**4) \
            - (sigma * (30 * a * rho + sigma * (-30 * mu + 7 * rho * sigma)) * (x_next - x)**3 * (v_next - v)) / (480 * (-1 + rho**2)**2 * v**4) \
            + ((540 * a * rho**2 + sigma * (-540 * mu * rho + (-97 + 160 * rho**2) * sigma)) * (x_next - x)**2 * (v_next - v)**2) / (2880 * (-1 + rho**2)**2 * v**4) \
            + ((-270 * a * rho * (-1 + 2 * rho**2) + sigma * (270 * mu * (-1 + 2 * rho**2) \
            + rho * (97 - 118 * rho**2) * sigma)) * (x_next - x) * (v_next - v)**3) / (1440*(-1 + rho**2)**2*sigma*v**4) \
            + ((360 * a * (-4 + 5 * rho**2) + sigma * (-360 * mu * rho * (-3 + 4 * rho**2) \
            + (-215 + 236 * rho**2) * sigma)) * (v_next - v)**4) / (5760 * (-1 + rho**2)**2 * sigma**2 * v**4) \
            + (sigma * (a * rho - mu * sigma) * (x_next - x)**3) / (24*(-1 + rho**2)**2*v**3) \
            + ((-3*a*rho**2 + sigma*(3*mu*rho + sigma - rho**2*sigma))* (x_next - x)**2*(v_next - v))/(24*(-1 + rho**2)**2*v**3) \
            + ((a*rho*(-7 + 10*rho**2) + sigma*(mu*(7 - 10*rho**2) + 2*rho*(-1 + rho**2)*sigma))*(x_next - x)* (v_next - v)**2)/(24*(-1 + rho**2)**2*sigma*v**3) \
            + ((a*(8 - 9*rho**2) + sigma*(mu*rho*(-7 + 8*rho**2) + sigma - rho**2*sigma))*(v_next - v)**3)/ (24*(-1 + rho**2)**2*sigma**2*v**3) \
            + (sigma**2*(x_next - x)**2)/(24*(-1 + rho**2)*v**2) + ((12*a + sigma*(-12*mu*rho + sigma))*(v_next - v)**2)/(24*(-1 + rho**2)*sigma**2*v**2) \
            - ((x_next - x)*(2*a*rho - 2*mu*sigma - 2*k*rho*v + sigma*v))/ (2*sigma*v - 2*rho**2*sigma*v) \
            + ((v_next - v)*(2*a - 2*mu*rho*sigma - 2*k*v + rho*sigma*v))/(2*sigma**2*v - 2*rho**2*sigma**2*v) \
            + ((6*a*rho - 6*mu*sigma + rho*sigma**2)*(x_next - x)*(v_next - v))/ (12*sigma*v**2 - 12*rho**2*sigma*v**2)

        c1 = (sigma*(a*rho - mu*sigma)*(x_next - x))/(12*(-1 + rho**2)*v**2) \
            + ((x_next - x)**2*(60*a**2*(1 + 2*rho**2) + 180*mu**2*sigma**2 \
            + 2*sigma**4 - 2*rho**2*sigma**4 - 60*a*sigma*(6*mu*rho + sigma - rho**2*sigma) \
            - 60*k**2*v**2 + 60*k*rho*sigma*v**2 - 15*sigma**2*v**2))/(2880*(-1 + rho**2)**2*v**3) \
            + ((v_next - v)*(-12*a**2 - 12*mu**2*sigma**2 + 4*mu*rho*sigma**3 - 2*sigma**4 + 2*rho**2*sigma**4 + 4*a*sigma*(6*mu*rho + (3 - 4*rho**2)*sigma) + 12*k**2*v**2 - 12*k*rho*sigma*v**2 + 3*sigma**2*v**2))/(48*(-1 + rho**2)*sigma**2*v**2) \
            + (1/(2880*(-1 + rho**2)**2*sigma**2*v**3))* ((v_next - v)**2*(180*a**2*(-3 + 4*rho**2) + 60*mu**2*(-7 + 10*rho**2)*sigma**2 - 240*mu*rho*(-1 + rho**2)*sigma**3 - 94*sigma**4 + 190*rho**2*sigma**4 - 96*rho**4*sigma**4 + 60*a*sigma*(mu*(14*rho - 20*rho**3) \
            + (9 - 23*rho**2 + 14*rho**4)*sigma) + 60*k**2*v**2 - 120*k**2*rho**2*v**2 - 60*k*rho*sigma*v**2 + 120*k*rho**3*sigma*v**2 + 15*sigma**2*v**2 - 30*rho**2*sigma**2*v**2)) + (1/(24*(-1 + rho**2)*sigma**2*v))* (12*a**2 + 12*mu**2*sigma**2 + 2*sigma**4 - 2*rho**2*sigma**4 \
            + 12*mu*(2*k*rho - sigma)*sigma* v + 12*k**2*v**2 - 12*k*rho*sigma*v**2 + 3*sigma**2*v**2 - 12*a*(2*mu*rho*sigma + sigma**2 - rho**2*sigma**2 + 2*k*v - rho*sigma*v)) + (1/(1440*(-1 + rho**2)**2*sigma*v**3))*((x_next - x)*(v_next - v)*(-60*a**2*(rho + 2*rho**3) \
            - 180*mu**2*rho*sigma**2 + 120*mu*(-1 + rho**2)*sigma**3 + 180*a*rho*sigma* (2*mu*rho + sigma - rho**2*sigma) \
            + rho*(2*(-1 + rho**2)*sigma**4 + 60*k**2*v**2 - 60*k*rho*sigma*v**2 + 15*sigma**2*v**2)))

        c2 = -((60*a**2*(-2 + rho**2) - 60*mu**2*sigma**2 + 23*(-1 + rho**2)*sigma**4 + 120*a*sigma*(mu*rho + sigma - rho**2*sigma))/(720*(-1 + rho**2)*v**2))

        lnpX = - torch.log(2 * delta_t * math.pi) - Dv + cm1/delta_t + c0 + delta_t * c1 + 0.5 * delta_t**2 * c2
        return torch.exp(lnpX)
    
    def euler_joint_density(self, s, v, s_next, v_next, delta_t):
        mu, k, theta, sigma, rho, _ = self.inv_reparam()

        mean_w = torch.log(v) + ((k * theta - 0.5 * sigma**2) / v - k) * delta_t     # w = log(v)
        std_w = sigma * torch.sqrt(delta_t / v)

        true_mean = theta + (v - theta) * torch.exp(- k * delta_t)
        true_var = v * sigma**2 * torch.exp( -k * delta_t) * (1 - torch.exp(-k * delta_t)) / k + theta * sigma**2 * (1 - torch.exp(-k * delta_t))**2 / (2 * k)

        # Moments matching for lognorm
        std_w = torch.sqrt(torch.log(1 + true_var / true_mean**2))
        mean_w = torch.log(true_mean) - 0.5 * std_w**2

        mean_x = torch.log(s) + (mu - 0.5 * v_next) * delta_t + (torch.sqrt(v_next * v) / sigma) * rho * (torch.log(v_next) - mean_w)
        std_x = torch.sqrt((1 - rho**2) * v_next * delta_t) * torch.ones_like(mean_x)

        log_transition = Normal(loc=mean_x, scale=std_x).log_prob(torch.log(s_next)) \
                    + Normal(loc=mean_w, scale=std_w).log_prob(torch.log(v_next))
        return torch.exp(log_transition)
