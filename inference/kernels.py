import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

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

    def update_sgd(self, data, model, particles):
        particles = self.particles.detach().clone().requires_grad_(True)

        optimizer = SGD(params=[particles], lr=self.lr)
        for _ in range(self.n_steps):
            optimizer.zero_grad()
            loss = - model.lpost(particles, data).sum()
            loss.backward()
            optimizer.step()

        return particles.detach()
