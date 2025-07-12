import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch import distributions as D
from torch.autograd import functional as F

class KernelSGLD():
    def __init__(self, n_steps, lr, lr_min, gamma):
        self.n_steps = n_steps
        self.lr = lr
        self.lr_min = lr_min
        self.gamma = gamma

    def polynomial_schedule(self, step):
        step = min(step, self.n_steps)
        decay = (1 - step / self.n_steps) ** self.gamma
        scaled = self.lr_min / self.lr + decay * (1 - self.lr_min / self.lr)
        return scaled

    def update(self, data, model, particles):
        particles = particles.detach().clone().requires_grad_(True)

        optimizer = SGD(params=[particles], lr=self.lr)
        group = optimizer.param_groups[0]
        schedule = lambda step: self.polynomial_schedule(step)
        scheduler = LambdaLR(optimizer, lr_lambda=schedule)
        for _ in range(self.n_steps):
            optimizer.zero_grad()
            loss = - model.lpost(particles, data).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

            eta = group['lr']
            preconditioner = 1.0
            noise_std = (2.0 * eta / preconditioner)**(0.5)

            particles.data.add_(torch.randn_like(particles) * noise_std)
        return particles.detach()

class KernelSGD():
    def __init__(self, n_steps, lr):
        self.n_steps = n_steps
        self.lr = lr

    def update(self, data, model, particles):
        particles = particles.detach().clone().requires_grad_(True)

        if model.is_sv:
            model.build_objective(data)
        optimizer = SGD(params=[particles], lr=self.lr)
        for _ in range(self.n_steps):
            optimizer.zero_grad()
            loss = - model.lpost(particles, data).sum()
            loss.backward()
            optimizer.step()

        return particles.detach()

class KernelMAP():
    def __init__(self, n_steps, lr):
        self.n_steps = n_steps
        self.lr = lr
        self.previous_map = None

    def compute_map(self, particle, objective):
        optimizer = Adam([particle], lr=self.lr)
        convergence_flag = False
        for i in range(self.n_steps):
            optimizer.zero_grad()
            loss = -objective(particle)
            loss.backward()

            grad_norm = torch.norm(particle.grad)
            if grad_norm < 0.5:
                convergence_flag = True
                break
            optimizer.step()
        if not convergence_flag:
            print(f"Convergence not reached.")
            print(f'Final grad: {grad_norm.item():.3f}')
        map = particle

        self.previous_map = map.squeeze()
        return map

    def update(self, data, model, particles):
        (n_particles, n_params) = particles.shape
        if self.previous_map is not None:
            init = self.previous_map.detach().unsqueeze(0).clone().requires_grad_(True)
        else:
            init = particles[0, :].detach().unsqueeze(0).clone().requires_grad_(True)
        objective = lambda theta: model.lpost(theta, data)
        map = self.compute_map(init, objective)

        tau = -F.hessian(objective, map).detach().squeeze()
        tau = (tau + tau.T) / 2 + 1e-6 * torch.eye(n_params)    # Ensures pos def

        new_particles = D.MultivariateNormal(
            loc=map.squeeze(),
            precision_matrix=tau
        ).sample(sample_shape=(n_particles,))
        return new_particles.detach()
