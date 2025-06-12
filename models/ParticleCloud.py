import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from svgd import compute_phi
from torchkde import KernelDensity

class KdePrior(KernelDensity):
    def __init__(self, particles, bandwidth):
        super().__init__(kernel='gaussian', bandwidth=bandwidth)
        self.fit(particles)

    def log_prob(self, eval):
        log_prob = self.score_samples(eval)
        return torch.relu(log_prob + 15.0) - 15.0

class ParticleCloud():
    def __init__(self, model, init_prior,
                 N: int, tau: float):
        self.model = model
        self.n_particles = N
        self.tau = tau

        init_samples = init_prior.sample(n_samples = self.n_particles)
        if self.n_particles == 1:
            init_samples = init_samples.unsqueeze(0)
        self.particles = init_samples.requires_grad_(True)
        self.prior = init_prior
        self.n_params = self.particles.shape[1]

    def get_particles(self):
        particles = self.particles.detach()
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
        return

    def update_prior(self):
        self.prior = KdePrior(particles=self.get_particles(), bandwidth='scott')

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

    def plot(self, background, **kwargs):
        particles = self.get_particles()

        fig, ax = plt.subplots(figsize=(5, 3))
        p = 50
        mus = torch.linspace(-1.0, 1.0, p)
        log_sigmas = torch.log(torch.linspace(0.01, 5.0, p))
        mus, log_sigmas = torch.meshgrid(mus, log_sigmas, indexing='ij')
        grid = torch.stack([mus.reshape(-1), log_sigmas.reshape(-1)], dim=1)

        if background == 'prior':
            background = self.prior.log_prob(grid).reshape(shape = (p, p))
        elif background == 'posterior':
            new_data = kwargs['new_data']
            background = self.log_posterior(grid, new_data).reshape(shape = (p, p))
        max = background.max().item()
        levels = torch.linspace(-10.0, max, 20)
        contour = ax.contourf(mus, log_sigmas, background, levels=levels, cmap='viridis', extend='min')
        fig.colorbar(contour, extend='min')

        ax.scatter(particles[:, 0], particles[:, 1], marker = '+', c='red', label='particles')

        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel(r"$\log\sigma$")
        return fig
