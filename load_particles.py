import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime
import pandas as pd
import matplotlib.dates as mdates

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
T = len(S)

bs_log_l = torch.load('logs/particles/bs_log_l.pt')
cev_log_l = torch.load('logs/particles/cev_log_l.pt')
nig_log_l = torch.load('logs/particles/nig_log_l.pt')
sv_log_l = torch.load('logs/particles/sv_log_l.pt')

optimization_freq = 20  # days
window = 200
decay_coef = 1.0
if window:
    start = window
else:
    start = 1
optimization_times = torch.arange(start + optimization_freq, T, step=optimization_freq)
n_grad_steps = 20

tau = 0.1
log_l_normalized = tau * torch.cat([bs_log_l, cev_log_l, nig_log_l, sv_log_l], dim=0)
log_normalization = torch.logsumexp(log_l_normalized, dim=0)
log_posterior = log_l_normalized - log_normalization
posterior = torch.exp(log_posterior)

fig, ax = plt.subplots(figsize=(10, 4))
plotting_index = np.array([0, *optimization_times])
plotting_times = dates[[0, *optimization_times]]
ax.plot(dates, 0.3 * S / S[0], label='S&P500', linestyle='--', linewidth=0.5)

n_particles = bs_log_l.shape[0]
for k in range(posterior.shape[0]):
    if k == 0:
        ax.plot(plotting_times, posterior[k, plotting_index], linewidth=0.5, marker='+', markersize=1, color='blue', label='bs')
    elif k==n_particles:
        ax.plot(plotting_times, posterior[k, plotting_index], linewidth=0.5, marker='+', markersize=1, color='red', label='cev')
    elif k==2 * n_particles:
        ax.plot(plotting_times, posterior[k, plotting_index], linewidth=0.5, marker='+', markersize=1, color='green', label='nig')
    elif k==3 * n_particles:
        ax.plot(plotting_times, posterior[k, plotting_index], linewidth=0.5, marker='+', markersize=1, color='pink', label='sv')
    elif k // n_particles == 0:
        ax.plot(plotting_times, posterior[k, plotting_index], linewidth=0.5, marker='+', markersize=1, color='blue')
    elif k // n_particles == 1:
        ax.plot(plotting_times, posterior[k, plotting_index], linewidth=0.5, marker='+', markersize=1, color='red')
    elif k // n_particles == 2:
        ax.plot(plotting_times, posterior[k, plotting_index], linewidth=0.5, marker='+', markersize=1, color='green')
    else:
        ax.plot(plotting_times, posterior[k, plotting_index], linewidth=0.5, marker='+', markersize=1, color='pink')

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
custom_grid_dates = dates[[0, *optimization_times]]

# Draw vertical grid lines at those custom positions
for date in custom_grid_dates:
    ax.axvline(date, color='gray', linestyle=':', linewidth=0.8, alpha=1.0)

ax.set_xlabel('time')
ax.set_ylabel('posterior probability')
fig.legend()
fig.tight_layout()
plt.savefig('figures/particles/particles.png')
