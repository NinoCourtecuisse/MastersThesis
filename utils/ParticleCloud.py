import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from utils.svgd import compute_phi
from utils.Prior import KdePrior

class ParticleCloud():
    def __init__(self, model, init_prior,
                 N: int, bandwidth='scott'):
        self.model = model
        self.n_particles = N
        self.bandwidth = bandwidth

        init_samples = init_prior.sample(n_samples = self.n_particles)
        if self.n_particles == 1:
            init_samples = init_samples.unsqueeze(0)
        self.particles = init_samples.requires_grad_(True)
        self.init_prior = init_prior
        self.prior = init_prior
        self.n_params = self.particles.shape[1]

    def get_particles(self):
        particles = self.particles.detach()
        return particles

    def log_posterior(self, eval, new_data):
        log_p = self.model.ll(eval, new_data) \
            + self.prior.log_prob(eval)
        return log_p

    def update_gd(self, new_data, n_steps, lr):
        for i in range(n_steps):
            #optimizer = Adam([self.particles], lr=lr)
            #optimizer.zero_grad()

            log_p = self.log_posterior(self.particles, new_data)
            grad_log_p = torch.autograd.grad(outputs=log_p.sum(), inputs=self.particles)[0]
            #self.particles.grad = - grad_log_p
            #optimizer.step()
            self.particles = self.particles + lr * grad_log_p
        return
    
    def update_sgld(self, new_data, n_steps, lr_1, lr_2, gamma):
        tmp = (lr_2 / lr_1)**(1 / gamma)
        b = (n_steps * tmp) / (1 - tmp)
        a = lr_1 * b**gamma
        for i in range(n_steps):
            lr = a * (b + i)**(- gamma)     # Polynomial decay

            log_p = self.log_posterior(self.particles, new_data)
            grad_log_p = torch.autograd.grad(outputs=log_p.sum(), inputs=self.particles)[0]

            self.particles = self.particles + lr * grad_log_p \
                            + torch.sqrt(torch.tensor(2 * lr)) * torch.randn_like(self.particles)
        return

    def update_prior(self, method='kde'):
        if method == 'kde':
            self.prior = KdePrior(particles=self.get_particles().clone(), bandwidth=self.bandwidth)
        elif method == 'initial':
            self.prior = self.init_prior
        else:
            raise NotImplementedError(f"Method {method} is not available.")

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
    
    def plot_bs(self, background, scale='optimization', **kwargs):
        particles = self.get_particles()

        fig, ax = plt.subplots(figsize=(5, 3))
        p = 50
        mus = torch.linspace(-0.5, 0.5, p)
        log_sigmas = torch.log(torch.linspace(0.01, 2.0, p))
        mus, log_sigmas = torch.meshgrid(mus, log_sigmas, indexing='ij')
        grid = torch.stack([mus.reshape(-1), log_sigmas.reshape(-1)], dim=1)

        if background == 'prior':
            background = self.prior.log_prob(grid).reshape(shape = (p, p))
        elif background == 'posterior':
            new_data = kwargs['new_data']
            background = self.log_posterior(grid, new_data).reshape(shape = (p, p))
        else:
            raise NotImplementedError(f"Background {background} is not available.")

        if scale == 'natural':
            grid_reparam = self.model.inverse_reparam(grid)
            x_vals = grid_reparam[:, 0].reshape(shape=(p,p))
            y_vals = grid_reparam[:, 1].reshape(shape=(p,p))
            particles = self.model.inverse_reparam(particles)
            ax.set_ylabel(r"$\sigma$")
        elif scale == 'optimization':
            x_vals = mus
            y_vals = log_sigmas
            ax.set_ylabel(r"$\log\sigma$")
        else:
            raise NotImplementedError(f"Scale {scale} is not available.")
        
        max = background.max().item()
        levels = torch.linspace(-10.0, max, 20)
        contour = ax.contourf(x_vals, y_vals, background, levels=levels, cmap='viridis', extend='min')
        fig.colorbar(contour, extend='min')
        if max > 0:
            fine_levels = torch.linspace(0.9 * max, max, 2)
        else:
            fine_levels = torch.linspace(1.1 * max, max, 2)
        #plt.contour(x_vals, y_vals, background, levels=fine_levels, cmap='viridis', extend='min')
        ax.scatter(particles[:, 0], particles[:, 1], marker = '+', c='red', label='particles')

        ax.set_xlabel(r"$\mu$")
        return fig

    def plot_cev(self, background, scale='optimization', mu=torch.tensor(0.0), **kwargs):
        particles = self.get_particles()

        fig, ax = plt.subplots(figsize=(5, 3))
        p = 50
        logdeltas = torch.log(torch.linspace(0.1, 10.0, p))
        logbetas = torch.log(torch.linspace(0.1, 3.0, p))
        logdeltas, logbetas = torch.meshgrid(logdeltas, logbetas, indexing='ij')
        grid = torch.stack([mu*torch.ones_like(logdeltas.reshape(-1)), logdeltas.reshape(-1), logbetas.reshape(-1)], dim=1)

        if background == 'prior':
            background = self.prior.log_prob(grid).reshape(shape = (p, p))
        elif background == 'posterior':
            new_data = kwargs['new_data']
            background = self.log_posterior(grid, new_data).reshape(shape = (p, p))
        else:
            raise NotImplementedError(f"Background {background} is not available.")

        if scale == 'natural':
            grid_reparam = self.model.inverse_reparam(grid)
            x_vals = grid_reparam[:, 1].reshape(shape=(p,p))
            y_vals = grid_reparam[:, 2].reshape(shape=(p,p))
            particles = self.model.inverse_reparam(particles)
            ax.set_ylabel(r"$\beta$")
        elif scale == 'optimization':
            x_vals = logdeltas
            y_vals = logbetas
            ax.set_ylabel(r"$\log\beta$")
        else:
            raise NotImplementedError(f"Scale {scale} is not available.")
        
        max = background.max().item()
        levels = torch.linspace(-10.0, max, 20)
        contour = ax.contourf(x_vals, y_vals, background, levels=levels, cmap='viridis', extend='min')
        fig.colorbar(contour, extend='min')
        if max > 0:
            fine_levels = torch.linspace(0.9 * max, max, 2)
        else:
            fine_levels = torch.linspace(1.1 * max, max, 2)
        #plt.contour(x_vals, y_vals, background, levels=fine_levels, cmap='viridis', extend='min')
        ax.scatter(particles[:, 1], particles[:, 2], marker = '+', c='red', label='particles')

        #ax.set_xlabel(r"$\delta$")
        return fig
