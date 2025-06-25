import torch
import math

class Ibis():
    def __init__(self, model, n_particles, kernel,
                 ess_rmin, window):
        self.model = model
        self.n_particles = n_particles
        self.kernel = kernel

        self.particles = None
        self.lweights = None
        self.marginal_ll = None

        self.ess_rmin = ess_rmin
        self.window = window
    
    def init_particles(self):
        samples = self.model.prior.sample(n_samples = self.n_particles)
        self.particles = self.model.transform.inv(samples)
        self.lweights = torch.zeros(self.n_particles)

    def update_marginal_ll(self, lw_increment):
        marginal_ll = torch.logsumexp(lw_increment, dim=0) \
                        - math.log(self.n_particles)
        self.marginal_ll = marginal_ll

    def get_marginal_ll(self):
        return self.marginal_ll

    def get_particles(self, parametrization='natural'):
        if parametrization == 'natural':
            return self.model.transform.to(self.particles.detach())
        else:
            return self.particles.detach()

    def compute_norm_weights(self):
        norm_weights = torch.exp(self.lweights - torch.logsumexp(self.lweights, dim=0))
        return norm_weights

    def compute_ess(self, norm_weights):
        ess = 1. / torch.sum(norm_weights ** 2)
        return ess

    def step(self, new_data, full_data):
        lw_increment = self.model.ll(self.particles, new_data)

        self.update_marginal_ll(lw_increment)
        self.lweights += lw_increment

        norm_weights = self.compute_norm_weights()
        ess = self.compute_ess(norm_weights)
        print(f"ESS={ess:.2f}")

        if ess < self.ess_rmin * self.n_particles:
            print("Resampling...")
            idx = torch.multinomial(norm_weights, self.n_particles, replacement=True)
            self.particles = self.particles[idx, :]
            self.lweights = torch.zeros(self.n_particles)

            self.particles = self.kernel.update(
                data=full_data[-self.window:],
                model=self.model,
                particles=self.particles,
            )

def backtest(model, kernel, batch_data, full_data,
             n_particles, ESS_rmin, window):

    ibis = Ibis(model, n_particles, kernel, ESS_rmin, window)
    ibis.init_particles()

    hist = {'particles':[], 'weights':[], 'll':[]}
    hist['particles'].append(ibis.get_particles())
    hist['weights'].append(ibis.compute_norm_weights())

    (n_batch, batch_size) = batch_data.shape
    for k in range(n_batch):
        print(f"Batch {k} / {n_batch}")
        batch = batch_data[k, :]
        t = (k + 1) * batch_size

        ibis.step(batch, full_data[:t])
        hist['particles'].append(ibis.get_particles())
        hist['weights'].append(ibis.compute_norm_weights())
        hist['ll'].append(ibis.get_marginal_ll())
    return hist
