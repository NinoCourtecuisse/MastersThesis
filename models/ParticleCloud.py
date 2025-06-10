import torch
import matplotlib.pyplot as plt

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
        return log_prior

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

    def plot(self, data):
        particles = self.get_particles().numpy()
        #self.prior = GmmPrior(centers = self.get_particles(), tau = self.tau)

        fig, ax = plt.subplots(figsize=(5, 3))

        x = torch.linspace(-1.0, 1.0, 50)
        y = torch.log(torch.linspace(1e-5, 1.1, 50))
        X, Y = torch.meshgrid(x, y, indexing='ij')
        result = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

        #out = self.prior.log_prob(self.model.inverse_reparam(result))
        out = self.log_posterior(result, data)
        out = out.reshape(shape = (50, 50))
        max = out.max().item()
        levels = torch.linspace(max * 0.5, max, 20).numpy()
        contour = ax.contourf(X, torch.exp(Y), out, levels=levels, cmap='viridis', extend='min')
        fig.colorbar(contour)

        fine_levels = torch.linspace(max * 0.98, max, 5)
        ax.contour(X, torch.exp(Y), out, levels=fine_levels, colors='k', linewidths=0.3)

        ax.scatter(particles[:, 0], particles[:, 1], marker = '+', c='red')

        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel(r"$\sigma$")
        return fig

    def gmm(self, eval: torch.tensor):
        new_eval = self.model.inverse_reparam(eval)
        previous_particles = self.model.inverse_reparam(self.previous_particles)
        tau = self.tau

        mean_sq = previous_particles.pow(2).sum(dim=1, keepdim=True)
        eval_sq = new_eval.pow(2).sum(dim=1, keepdim=True)
        cross_term = previous_particles @ new_eval.T
        sq_dists = mean_sq + eval_sq.T - 2 * cross_term
        gaussian_eval = (2 * torch.pi * tau**2)**(- self.n_params / 2) \
                        * torch.exp(- 0.5 * sq_dists / (tau ** 2))
        prior = torch.mean(gaussian_eval, dim = 0)
        log_prior = prior.log()
        return log_prior

    def log_posterior(self, eval, new_data):
        log_p = self.model.ll(eval, new_data) \
            + self.prior.log_prob(self.model.inverse_reparam(eval))
        return log_p

    def update_gd(self, new_data, n_steps, lr):  # TODO: Can we use adam?
        for i in range(n_steps):
            log_p = self.log_posterior(self.particles, new_data)
            grad_log_p = torch.autograd.grad(outputs=log_p.sum(), inputs=self.particles)[0]
            self.particles = self.particles + lr * grad_log_p

        self.prior = GmmPrior(centers = self.get_particles(), tau = self.tau)
        return

    def update_svgd(self, new_data, n_steps, lr):
        return NotImplementedError
