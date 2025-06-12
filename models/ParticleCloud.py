import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from svgd import compute_phi

class GmmPrior:
    def __init__(self, centers, tau):
        # Expects centers in natural parametrization
        self.centers = centers
        self.tau = tau
        self.d = centers.shape[1]
    
    def log_prob(self, eval):
        # Expects eval in natural parametrization
        centers = self.centers
        tau = self.tau

        mean_sq = centers.pow(2).sum(dim=1, keepdim=True)
        eval_sq = eval.pow(2).sum(dim=1, keepdim=True)
        cross_term = centers @ eval.T
        sq_dists = mean_sq + eval_sq.T - 2 * cross_term
        gaussian_eval = (2 * torch.pi * tau**2)**(- self.d / 2) \
                        * torch.exp(- 0.5 * sq_dists / (tau ** 2))
        prior = torch.mean(gaussian_eval, dim = 0)
        log_prior = prior.log()
        return torch.relu(log_prior + 10**4) - 10**4    # Clamp to low value to avoid -inf

class ParticleCloud():
    def __init__(self, model, init_prior,
                 N: int, tau: float):
        self.model = model
        self.n_particles = N
        self.tau = tau

        init_samples = init_prior.sample(n_samples = self.n_particles)
        if self.n_particles == 1:
            init_samples = init_samples.unsqueeze(0)
        self.particles = self.model.reparam(init_samples).requires_grad_(True)
        self.prior = init_prior
        self.n_params = self.particles.shape[1]

    def get_particles(self):
        # Get the particles in natural parametrization
        particles = self.model.inverse_reparam(self.particles.detach())
        return particles

    def log_posterior(self, eval, new_data):
        log_p = self.model.ll(eval, new_data) / len(new_data) \
            + self.prior.log_prob(self.model.inverse_reparam(eval))
        return log_p

    def update_gd(self, new_data, n_steps, lr):  # TODO: Can we use adam?
        for i in range(n_steps):
            log_p = self.log_posterior(self.particles, new_data)
            grad_log_p = torch.autograd.grad(outputs=log_p.sum(), inputs=self.particles)[0]
            self.particles = self.particles + lr * grad_log_p

        self.prior = GmmPrior(centers = self.get_particles(), tau = self.tau)
        return

    def update_prior(self):
        self.prior = GmmPrior(centers = self.get_particles(), tau = self.tau)

    def update_svgd(self, new_data, n_steps, lr, h):
        for i in range(n_steps):
            optimizer = Adam([self.particles], lr=lr)
            optimizer.zero_grad()

            # Construct G
            log_p = self.log_posterior(self.particles, new_data)
            G = torch.autograd.grad(outputs=log_p.sum(), inputs=self.particles)[0]

            with torch.no_grad():
                phi = compute_phi(G, self.particles, h, h_per_dimension=False)
            self.particles.grad = - phi
            optimizer.step()
        return

    def plot_prior(self):
        particles = self.get_particles()

        fig, ax = plt.subplots(figsize=(5, 3))
        p = 50
        mus = torch.linspace(-1.0, 1.0, p)
        sigmas = torch.linspace(0.01, 5.0, p)
        mus, sigmas = torch.meshgrid(mus, sigmas, indexing='ij')
        grid = torch.stack([mus.reshape(-1), sigmas.reshape(-1)], dim=1)

        prior = self.prior.log_prob(grid).reshape(shape = (p, p))
        max = prior.max().item()
        levels = torch.linspace(-10.0, max, 20)
        contour = ax.contourf(mus, torch.log(sigmas), prior, levels=levels, cmap='viridis', extend='min')
        fig.colorbar(contour, extend='min')

        ax.scatter(particles[:, 0], torch.log(particles[:, 1]), marker = '+', c='red', label='particles')

        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel(r"$\log\sigma$")
        return fig

    def plot_posterior(self, new_data):
        particles = self.get_particles()

        fig, ax = plt.subplots(figsize=(5, 3))
        p = 50
        mus = torch.linspace(-1.0, 1.0, p)
        log_sigmas = torch.log(torch.linspace(0.01, 5.0, p))
        mus, log_sigmas = torch.meshgrid(mus, log_sigmas, indexing='ij')
        grid = torch.stack([mus.reshape(-1), log_sigmas.reshape(-1)], dim=1)

        posterior = self.log_posterior(grid, new_data).reshape(shape = (p, p))
        max = posterior.max().item()
        levels = torch.linspace(-10.0, max, 20)
        contour = ax.contourf(mus, log_sigmas, posterior, levels=levels, cmap='viridis', extend='min')
        fig.colorbar(contour, extend='min')

        ax.scatter(particles[:, 0], torch.log(particles[:, 1]), marker = '+', c='red')

        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel(r"$\log\sigma$")
        return fig
