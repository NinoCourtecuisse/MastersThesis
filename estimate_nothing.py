import logging
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import torch

from models.bs_class import Bs
from models.cev_class import Cev

logging.basicConfig(
    filename='logs/estimate_nothing/estimate_nothing.log',
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


# STEP 1: Determine fixed set of parameters
#   Calibrate each models every months (compute MLE)
dt = torch.tensor(1/252, requires_grad=False)
window = 200
decay_coef = 1.0
T = len(S)
n_models = 4
optimization_freq = 20  #days
optimization_times = torch.arange(window + optimization_freq, T, step=optimization_freq)
n_grad_steps = 50

def compute_mle(optimization_times, model, optimizer, verbose=False):
    n_params = sum(p.numel() for p in model.parameters())
    mles = torch.zeros(size=(len(optimization_times), n_params))
    for i in range(len(optimization_times)):
        t = optimization_times[i]
        if verbose and t % 100 == 0: print(t.item())

        for _ in range(n_grad_steps):
            optimizer.zero_grad()
            loss = - model(S, t=t, delta_t=dt, decay_coef=decay_coef, window=window)
            loss.backward()
            optimizer.step()

        params = model.get_params()
        mles[i] = torch.tensor([p for p in params.values()])
    return mles
        
mu = torch.tensor(0.01)
sigma = torch.tensor(0.2)
bs_model = Bs(mu, sigma)
bs_optimizer = torch.optim.Adam(bs_model.parameters(), lr=0.1)
bs_mles = compute_mle(optimization_times, bs_model, bs_optimizer, verbose=False)
logging.info(f"bs mles: \n {bs_mles}")

mu = torch.tensor(0.01)
delta = torch.tensor(10.0)
beta = torch.tensor(1.0)
cev_model = Cev(mu, delta, beta)
cev_optimizer = torch.optim.Adam(cev_model.parameters(), lr=0.1)
cev_mles = compute_mle(optimization_times, cev_model, cev_optimizer, verbose=False)
logging.info(f"cev mles: \n {cev_mles}")

# Span 5 - 10 points between the lowest and highest values
bs_min = torch.min(bs_mles, dim=0)[0]
bs_max = torch.max(bs_mles, dim=0)[0]
w = torch.linspace(0, 1, 10).unsqueeze(1)
bs_params = (1 - w) * bs_min + w * bs_max
bs_params = torch.cartesian_prod(bs_params[:, 0], bs_params[:, 1])


cev_min = torch.min(cev_mles, dim=0)[0]
cev_max = torch.max(cev_mles, dim=0)[0]
w = torch.linspace(0, 1, 8).unsqueeze(1)
cev_params = (1 - w) * cev_min + w * cev_max
cev_params = torch.cartesian_prod(cev_params[:, 0], cev_params[:, 1], cev_params[:, 2])


# STEP 2: Compute the likelihood of each parameters
with torch.no_grad():
    bs_log_l = torch.zeros(size=(len(bs_params), T))
    for i in range(len(bs_params)):
        if i % 20 == 0: print(i)
        model = Bs(*bs_params[i])
        for t in range(window, T):
            bs_log_l[i, t] = model.forward(S, t=t, delta_t=dt, decay_coef=decay_coef, window=window)


with torch.no_grad():
    cev_log_l = torch.zeros(size=(len(cev_params), T))
    for i in range(len(cev_params)):
        if i % 20 == 0: print(i)
        model = Cev(*cev_params[i])
        for t in range(window, T):
            cev_log_l[i, t] = model.forward(S, t=t, delta_t=dt, decay_coef=decay_coef, window=window)


# Step 3: Select only parameters which where the best of their class at some point
bs_best_idx = torch.unique(torch.max(bs_log_l[:, window:], dim=0)[1])
cev_best_idx = torch.unique(torch.max(cev_log_l[:, window:], dim=0)[1])

print(len(bs_best_idx))
print(len(cev_best_idx))

"""
tmp = torch.max(bs_log_l[:, window:], dim=0)[0] - bs_log_l[:, window:]
epsilon = 0.01
mask = torch.where(tmp <= epsilon, 1.0, 0.0)
bs_indices = torch.nonzero(mask.any(dim=1), as_tuple=False).squeeze()
n_params = len(bs_indices)
print(n_params)
"""

if len(bs_best_idx) > len(cev_best_idx):
    bs_best_idx = bs_best_idx[:len(cev_best_idx)]
elif len(cev_best_idx) > len(bs_best_idx):
    cev_best_idx = cev_best_idx[:len(bs_best_idx)]
print(len(bs_best_idx))
print(len(cev_best_idx))

bs_indices = bs_best_idx
cev_indices = cev_best_idx
n_params = len(bs_indices)

import matplotlib.dates as mdates
import numpy as np

plotting_index = np.array([0, *optimization_times])
plotting_times = dates[[0, *optimization_times]]

fig, ax = plt.subplots(figsize=(10, 4))
custom_grid_dates = dates[[0, *optimization_times]]
custom_grid_idx = [0, *optimization_times]
for idx in custom_grid_dates:
    ax.axvline(idx, color='gray', linestyle=':', linewidth=0.8, alpha=1.0)

tau = 0.01
ax.plot(plotting_times, 10 * S[plotting_index] / S[0], linewidth=0.5, c='grey')
for idx in bs_indices:
    ax.plot(plotting_times, tau * bs_log_l[idx, plotting_index], linewidth=0.8, c='blue', alpha=0.3)
for idx in cev_indices:
    ax.plot(plotting_times, tau * cev_log_l[idx, plotting_index], linewidth=0.8, c='red', alpha=0.3)
ax.grid()
ax.legend()
plt.savefig('figures/estimate_nothing/en_likelihoods.png')

log_l_normalized = tau * torch.cat([bs_log_l[bs_indices, :], cev_log_l[cev_indices, :]], dim=0)
log_normalization = torch.logsumexp(log_l_normalized, dim=0)
log_posterior = log_l_normalized - log_normalization
posterior = torch.exp(log_posterior)

fig, ax = plt.subplots(figsize=(10, 4))
custom_grid_dates = dates[[0, *optimization_times]]
for date in custom_grid_dates:
    ax.axvline(date, color='gray', linestyle=':', linewidth=0.8, alpha=1.0, marker='+')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax.plot(dates, 0.1 * S / S[0], linewidth=0.5, c='grey')
for i in range(n_params):
    ax.plot(plotting_times, posterior[i, plotting_index], linewidth=0.3, c='blue', alpha=0.8)
for i in range(n_params, 2 * n_params):
    ax.plot(plotting_times, posterior[i, plotting_index], linewidth=0.3, c='red', alpha=0.8)
ax.grid()
ax.legend()
plt.savefig('figures/estimate_nothing/en_posteriors.png')

fig, ax = plt.subplots(figsize=(10, 4))
custom_grid_dates = dates[[0, *optimization_times]]
for date in custom_grid_dates:
    ax.axvline(date, color='gray', linestyle=':', linewidth=0.8, alpha=1.0)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax.plot(dates, 0.1 * S / S[0], linewidth=0.5, c='grey')
bs_posterior = torch.sum(posterior[:n_params, :], dim=0)
cev_posterior = torch.sum(posterior[n_params:2*n_params, :], dim=0)
ax.plot(plotting_times, bs_posterior[plotting_index], label='bs', linewidth=0.8, c='blue', alpha=0.5)
ax.plot(plotting_times, cev_posterior[plotting_index], label='cev', linewidth=0.8, c='red', alpha=0.5)

ax.grid()
ax.legend()
plt.savefig('figures/estimate_nothing/en_marginal_posteriors.png')