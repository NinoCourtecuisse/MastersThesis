import logging
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import torch
from time import time

from models.bs_class import Bs
from models.cev_class import Cev
from models.nig import Nig
from bayes.prior import Prior

from torch.distributions import Uniform
torch.manual_seed(42)

logging.basicConfig(
    filename='logs/particles.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Load data
start = datetime.datetime.strptime('2006-01-03', '%Y-%m-%d')
end = datetime.datetime.strptime('2012-12-31', '%Y-%m-%d')

SPOT_PATH = 'data/spx_spot.csv'
spot_data = pd.read_csv(SPOT_PATH, sep=',')
spot_data['date'] = pd.to_datetime(spot_data['date'])
spot_data = spot_data[(spot_data['date'] >= start) & (spot_data['date'] <= end)]
spot_data.set_index('date', inplace=True)

S = spot_data['close'].to_numpy()
dates = spot_data.index.to_numpy()
S = torch.tensor(S, dtype=torch.float32)


# Bayesian setup
dt = torch.tensor(1/252, requires_grad=False)
window = None
decay_coef = 0.99
T = len(S)
n_models = 3
log_l = torch.zeros(size=(n_models, T))
optimization_freq = 20  # days
if window:
    start = window
else:
    start = 1
optimization_times = torch.arange(start + optimization_freq, T, step=optimization_freq)
n_grad_steps = 20


# Black-Scholes particles
bs_prior = Prior(dists=[
    Uniform(-1.0, 1.0),
    Uniform(0.1, 1.5)
])
tic = time()
n_particles = 20
logging.info(f"Number of BS particles: {n_particles}")
bs_log_l = torch.zeros(size=(n_particles, T))
for i in range(n_particles):
    particle = bs_prior.sample()
    model = Bs(particle[0], particle[1])
    params = model.get_params()
    logging.info(f"{i} / {n_particles}")
    logging.info(f"t=0 mu={params['mu']:.3f}, sigma={params['sigma']:.3f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    bs_log_l[i, :] = model.likelihood_with_updates(optimizer, optimization_times, n_grad_steps, \
                                                   S, dt, decay_coef, window, start, \
                                                    logging=logging, verbose=False)
tac = time()
logging.info(f'Elapsed time: {tac - tic:.3f}')
torch.save(bs_log_l, f='logs/bs_log_l.pt')

# CEV particles
cev_prior = Prior([
    Uniform(-1.0, 1.0),
    Uniform(0.1, 10.0),
    Uniform(0.1, 1.9)
])
tic = time()
n_particles = 20
logging.info(f"Number of CEV particles: {n_particles}")
cev_log_l = torch.zeros(size=(n_particles, T))
for i in range(n_particles):
    particle = cev_prior.sample()
    model = Cev(particle[0], particle[1], particle[2])
    params = model.get_params()
    logging.info(f"{i} / {n_particles}")
    logging.info(f"t=0 mu={params['mu']:.3f}, delta={params['delta']:.3f}, beta={params['beta']:.3f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    cev_log_l[i, :] = model.likelihood_with_updates(optimizer, optimization_times, n_grad_steps, \
                                                   S, dt, decay_coef, window, start, \
                                                    logging=logging, verbose=False)
tac = time()
logging.info(f'Elapsed time: {tac - tic:.3f}')
torch.save(cev_log_l, f='logs/cev_log_l.pt')


# NIG particles
nig_prior = Prior([
    Uniform(-1.0, 1.0),
    Uniform(0.1, 1.5),
    Uniform(-3.0, 0.0),
    Uniform(0.001, 0.1)
])
tic = time()
n_particles = 20
logging.info(f"Number of CEV particles: {n_particles}")
nig_log_l = torch.zeros(size=(n_particles, T))
for i in range(n_particles):
    particle = nig_prior.sample()
    model = Nig(particle[0], particle[1], particle[2], particle[3])
    params = model.get_params()
    logging.info(f"{i} / {n_particles}")
    logging.info(f"t=0 mu={params['mu']:.3f}, sigma={params['sigma']:.3f}, xi={params['xi']:.3f}, eta={params['eta']:.3f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    nig_log_l[i, :] = model.likelihood_with_updates(optimizer, optimization_times, n_grad_steps, \
                                                   S, dt, decay_coef, window, start, \
                                                    logging=logging, verbose=False)
tac = time()
logging.info(f'Elapsed time: {tac - tic:.3f}')
torch.save(nig_log_l, f='logs/nig_log_l.pt')
