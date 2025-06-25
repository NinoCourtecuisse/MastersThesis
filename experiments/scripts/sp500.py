import torch
from torch import distributions as D
import matplotlib.pyplot as plt

from utils.data import load_data, batch_data
from utils.special_functions import sliding_sum
from utils.priors import IndependentPrior

from inference.ibis import backtest
from inference import KernelSGLD
from models import Bs, Cev, Nig

path = 'data/spx_spot.csv'
dates, s = load_data(path, start='2006-01-03', end='2012-12-31')

batch_size = 15
s_batch = batch_data(s, batch_size=batch_size)
print(f"Batched dataset shape: {s_batch.shape}")

torch.manual_seed(0)
dt = torch.tensor(1 / 252)
ESS_rmin = 0.5
window = 100
kernel = KernelSGLD(n_steps=1000, lr=1e-3, lr_min=1e-4, gamma=1.0)

########## Black-Scholes ##########
n_particles = 50
bs_prior = IndependentPrior([
    D.Normal(0.0, 0.1),     # mu
    D.Uniform(0.01, 1.5)    # sigma
])
bs_model = Bs(dt, bs_prior)
bs_hist = backtest(bs_model, kernel, s_batch, s, n_particles, ESS_rmin, window)

########## CEV ##########
n_particles = 50
cev_prior = IndependentPrior([
    D.Normal(0.0, 0.1),         # mu
    D.Uniform(0.01, 200.0),     # delta
    D.Uniform(0.01, 2.5)        # beta
])
cev_model = Cev(dt, cev_prior)
cev_hist = backtest(cev_model, kernel, s_batch, s, n_particles, ESS_rmin, window)

########## NIG ##########
n_particles = 50
nig_prior = IndependentPrior([
    D.Normal(0.0, 0.1),       # mu
    D.Uniform(0.01, 1.5),     # sigma
    D.Uniform(-3.0, 0.0),     # xi
    D.Uniform(0.001, 0.1)     # eta
])
nig_model = Nig(dt, nig_prior)
nig_hist = backtest(nig_model, kernel, s_batch, s, n_particles, ESS_rmin, window)

# Model prior
model_prior = torch.ones(size=(3,)) / 3

bs_post = sliding_sum(torch.tensor(bs_hist['ll']), w=window // batch_size) + model_prior[0].log()
cev_post = sliding_sum(torch.tensor(cev_hist['ll']), w=window // batch_size) + model_prior[1].log()
nig_post = sliding_sum(torch.tensor(nig_hist['ll']), w=window // batch_size) + model_prior[2].log()

tmp = torch.stack([bs_post, cev_post, nig_post])
lnorm_constant = torch.logsumexp(tmp, dim=0)
bs_lpost = bs_post - lnorm_constant
cev_lpost = cev_post - lnorm_constant
nig_lpost = nig_post - lnorm_constant

plt.figure(figsize=(5, 3))
plt.plot(bs_hist['ll'], label='bs')
plt.plot(cev_hist['ll'], label='cev')
plt.plot(nig_hist['ll'], label='nig')
plt.grid()
plt.legend()
plt.title('Model marginal log-likelihood')
plt.show()

plt.figure(figsize=(5, 3))
plt.plot(bs_lpost, label='bs')
plt.plot(cev_lpost, label='cev')
plt.plot(nig_lpost, label='nig')
plt.grid()
plt.legend()

plt.title('Model log posterior')
plt.show()