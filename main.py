import logging
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import torch
from time import time
from scipy.special import softmax

from models.sabr_class import Sabr
from models.heston_class import Heston
from models.bs_class import Bs
from models.cev_class import Cev
from bayes.utils import likelihood_with_updates

logging.basicConfig(
    filename='logs/likelihood.log',
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
window = 200
T = len(S)
n_models = 4
log_l = torch.zeros(size=(n_models, T))
optimization_freq = 20  #days
optimization_times = torch.arange(optimization_freq, len(S), step=optimization_freq)
n_grad_steps = 50

# Bayesian analysis

logging.info(f"Model: BS")
mu = torch.tensor(0.01)
sigma = torch.tensor(0.2)
bs_model = Bs(mu, sigma)
mu, sigma = bs_model.inv_reparam()
logging.info(f"Init params: mu: {mu.item():.3f}, sigma: {sigma.item():.3f}")

tic = time()
optimizer = torch.optim.Adam(bs_model.parameters(), lr=0.1)
log_l[0, :] = likelihood_with_updates(bs_model, optimizer, optimization_times, n_grad_steps, S, window, logging)
tac = time()
logging.info(f'Elapsed time: {tac - tic:.3f}')

logging.info(f"Model: CEV")
mu = torch.tensor(0.01)
delta = torch.tensor(10.0)
beta = torch.tensor(1.0)
cev_model = Cev(mu, delta, beta)
mu, delta, beta = cev_model.inv_reparam()
logging.info(f"Init params: mu: {mu.item():.3f}, delta: {delta.item():.3f}, beta: {beta.item():.3f}")

tic = time()
optimizer = torch.optim.Adam(cev_model.parameters(), lr=0.1)
log_l[1, :] = likelihood_with_updates(cev_model, optimizer, optimization_times, n_grad_steps, S, window, logging)
tac = time()
logging.info(f'Elapsed time: {tac - tic:.3f}')

logging.info(f"Model: Heston")
mu = torch.tensor(0.01)
k = torch.tensor(1.0)
theta = torch.tensor(0.04)
sigma = torch.tensor(0.5)
rho = torch.tensor(-0.3)
v0 = torch.tensor(0.04)
heston_model = Heston(mu, k, theta, sigma, rho, v0)
mu, k, theta, sigma, rho, v0 = heston_model.inv_reparam()
logging.info(f"Init params: mu: {mu.item():.3f}, k: {k.item():.3f}, theta: {theta.item():.3f}, sigma: {sigma.item():.3f}, rho: {rho.item():.3f}, v0: {v0.item():.3f}")

tic = time()
optimizer = torch.optim.Adam(heston_model.parameters(), lr=0.1)
log_l[2, :] = likelihood_with_updates(heston_model, optimizer, optimization_times, n_grad_steps, S, window, logging)
tac = time()
logging.info(f'Elapsed time: {tac - tic:.3f}')

logging.info(f"Model: Sabr")
mu = torch.tensor(-0.01)
beta = torch.tensor(1.0)
sigma = torch.tensor(0.5)
rho = torch.tensor(-0.8)
delta_0 = torch.tensor(10.0)
sabr_model = Sabr(mu, beta, sigma, rho, delta_0)
mu, beta, sigma, rho, delta_0 = sabr_model.inv_reparam()
logging.info(f"Init params: mu: {mu.item():.3f}, beta: {beta.item():.3f}, sigma: {sigma.item():.3f}, rho: {rho.item():.3f}, delta_0: {delta_0.item()}")

tic = time()
optimizer = torch.optim.Adam(sabr_model.parameters(), lr=0.1)
log_l[3, :] = likelihood_with_updates(sabr_model, optimizer, optimization_times, n_grad_steps, S, window, logging)
tac = time()
logging.info(f'Elapsed time: {tac - tic:.3f}')

import matplotlib.dates as mdates
import numpy as np

plotting_index = np.array([0, *optimization_times])
plotting_times = dates[[0, *optimization_times]]

fig, ax = plt.subplots(figsize=(10, 4))
custom_grid_dates = dates[[0, *optimization_times]]
custom_grid_idx = [0, *optimization_times]
for idx in custom_grid_idx:
    ax.axvline(idx, color='gray', linestyle=':', linewidth=0.8, alpha=1.0)
#ax.xaxis.set_major_locator(mdates.YearLocator())
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

tau = 0.01
log_l_normalized = tau * log_l
ax.plot(10 * S / S[0], linewidth=0.5, c='grey')
ax.plot(log_l_normalized[0, :], label='bs', linewidth=0.8)
ax.plot(log_l_normalized[1, :], label='cev', linewidth=0.8)
ax.plot(log_l_normalized[2, :], label='heston', linewidth=0.8)
ax.plot(log_l_normalized[3, :], label='sabr', linewidth=0.8)
ax.grid()
ax.legend()
plt.savefig('figures/likelihoods.png')

fig, ax = plt.subplots(figsize=(10, 4))
custom_grid_dates = dates[[0, *optimization_times]]
for date in custom_grid_dates:
    ax.axvline(date, color='gray', linestyle=':', linewidth=0.8, alpha=1.0)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

#posterior = softmax(log_l_normalized, axis=0)
log_normalization = torch.logsumexp(log_l_normalized, dim=0)
log_posterior = log_l_normalized - log_normalization
posterior = torch.exp(log_posterior)
ax.plot(dates, 0.5 * S / S[0], linewidth=0.5, c='grey')
ax.plot(dates, posterior[0, :], label='bs', linewidth=0.8)
ax.plot(dates, posterior[1, :], label='cev', linewidth=0.8)
ax.plot(dates, posterior[2, :], label='heston', linewidth=0.8)
ax.plot(dates, posterior[3, :], label='sabr', linewidth=0.8)
ax.grid()
ax.legend()
plt.savefig('figures/posteriors.png')
