import argparse

import torch
from torch import distributions as D
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from src.utils.data import load_data
from src.utils.priors import IndependentPrior, CevPrior
from src.utils.distributions import ScaledBeta
from src.inference.estimateNothing import compute_log_weights, update_particles

from src.models import Bs, Cev, Nig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()

def main(args):
    torch.manual_seed(args.seed)

    ######## Load data ########
    path = 'data/spx_spot.csv'
    dates, S = load_data(path, start='2006-01-01', end='2024-01-01')

    ######## Hyperparameters ########
    dt = torch.tensor(1 / 252)
    N = 100    # Number of particles
    T = len(S) - 1
    batch_size = 50
    first_update = 100
    update_freq = 20
    update_dates = torch.arange(start=first_update, end=T+1, step=update_freq) # Wait 100 days of data for first update, then every 40 days.
    n_updates = len(update_dates)
    window = 252

    ######## Instantiate the model classes ########
    bs_prior = IndependentPrior([
        D.Normal(0., 0.05),
        ScaledBeta(2.0, 2.0, low=0.01, high=1.0)
    ])
    bs_model = Bs(dt, bs_prior)
    bs_particles = bs_model.transform.inv(bs_prior.sample(n_samples=N))

    cev_prior = CevPrior(
        mu_dist = D.Normal(0., 0.05),
        beta_dist = ScaledBeta(5., 5., low=torch.tensor(0.5), high=torch.tensor(2.0)),
        v = 0.2, S=1000
    )
    cev_model = Cev(dt, cev_prior)
    cev_particles = cev_model.transform.inv(cev_prior.sample(n_samples=N))

    nig_prior = IndependentPrior([
        D.Normal(0., 0.05),
        ScaledBeta(2.0, 2.0, low=0.01, high=1.0),
        D.Normal(-2.0, 2.0),
        ScaledBeta(1.5, 5.0, low=1e-6, high=0.1)
    ])
    nig_model = Nig(dt, nig_prior)
    nig_particles = nig_model.transform.inv(nig_prior.sample(n_samples=N))

    ######## Proposed method ########
    model_classes = [bs_model, cev_model, nig_model]
    M = len(model_classes)

    particles = [bs_particles, cev_particles, nig_particles]
    init_model_classes_log_prior = torch.log(torch.ones(size=(M,)) / M) # Uniform
    lrs = [1e-4, 1e-4, 1e-3]
    model_classes_log_prior = init_model_classes_log_prior

    n_params = [len(p[0, :]) for p in particles]    # Number of parameters per model (2 for Black-Scholes, 3 for CEV, ...)

    # Save for plot: History of all particles, Estimate Nothing weights, prior on model classes
    hist_particles = [torch.zeros(size=(n_updates+2, N, d)) for d in n_params]
    for m in range(M):
        hist_particles[m][0, :, :] = model_classes[m].transform.to(particles[m])
    hist_log_pi = [torch.zeros(size=(T, N)) for m in range(M)]
    hist_ll = [torch.zeros(size=(T, N)) for m in range(M)]
    hist_log_prior = [torch.zeros(size=(T+1,)) for m in range(M)]
    for m in range(M):
        hist_log_prior[m][0] = model_classes_log_prior[m]
    for i in range(n_updates + 1):
        print(f"Update index: {i+1} / {n_updates + 1}")
        if i==0:
            t_k = update_dates[i]
            t_k_minus_1 = 0
        elif i==n_updates:
            t_k = T
            t_k_minus_1 = update_dates[i - 1]
        else:
            t_k = update_dates[i]
            t_k_minus_1 = update_dates[i - 1]

        # Step 1: Follow "Estimate Nothing"
        log_pi, log_iw, ll = compute_log_weights(
            model_classes, particles, model_classes_log_prior, price_data=S[t_k_minus_1:t_k+1],  # Use the data from previous update up to and including t_k
            temperature=1.
        )
        # Step 2: Update the particles
        # 2.1: Update the prior on model classes
        if i == 0:
            model_classes_log_prior = model_classes_log_prior
        else:
            beta = 0.8
            model_classes_log_prior = beta * torch.logsumexp(log_pi[:, -1, :], dim=1) + (1 - beta) * init_model_classes_log_prior
            model_classes_log_prior -= torch.logsumexp(model_classes_log_prior, dim=0)

        # 2.2 and 2.3: Resample and move 
        #   -> Only consider recent past (fixed window) for the update. Not Bayesian exact but better results.
        #dataset = TensorDataset(S[:t_k+1])
        dataset = TensorDataset(S[max(t_k+1-window, 0):t_k+1])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        n_batch = len(dataloader)
        #batch_weights = torch.exp(-(n_batch - torch.arange(n_batch)))
        #batch_weights /= torch.sum(batch_weights)
        batch_weights = torch.ones(size=(n_batch,)) / n_batch

        new_particles = update_particles(
            model_classes, particles, log_iw, dataloader, batch_weights,
            n_grad_steps=100, lrs=lrs,
            resample=True, move=True, verbose=False
        )
        particles = new_particles

        # Save
        for m in range(M):
            hist_particles[m][i+1, :, :] = model_classes[m].transform.to(particles[m])
            hist_log_pi[m][t_k_minus_1:t_k] = log_pi[m]
            hist_ll[m][t_k_minus_1:t_k] = ll[m]
            hist_log_prior[m][t_k_minus_1+1:t_k] = torch.logsumexp(log_pi[m, :-1, :], dim=1)
            hist_log_prior[m][t_k] = model_classes_log_prior[m]

    ######## Plot results ########
    colors = ['blue', 'red', 'green']

    params_names = [['\\mu', '\\sigma'], ['\\mu', '\\delta', '\\beta'], ['\\mu', '\\sigma', '\\xi', '\\eta']]
    for m in range(M):
        fig, axes = plt.subplots(n_params[m], 1, figsize=(10, 4), sharex=True)
        param_str = ", ".join(params_names[m])
        title = (
            f"Set of parameters considered for model class {model_classes[m].__class__.__name__}\n"
            f"${param_str}$ from top to bottom."
        )
        axes[0].set_title(title)
        for i in range(n_params[m]):
            ax = axes[i]
            for n in range(N):
                ax.scatter(dates[update_dates], hist_particles[m][1:-1, n, i], marker='+', s=10.0)
            ax.grid()
        fig.tight_layout()

    fig, ax = plt.subplots(figsize=(10, 4))
    for m in range(M):
        for i in range(n_params[m]):
            ax.plot(dates[update_dates[0]+1:], hist_ll[m][update_dates[0]:], linewidth=0.3, c=colors[m])
    ax.vlines(x=dates[update_dates], ymin=0, ymax=1, label='update', colors='black', linestyle='--', linewidth=0.5)
    #ax.plot(dates[update_dates[0]:], torch.log(S[update_dates[0]:] / S[update_dates[0]-1:-1]) + 0.25, linewidth=0.5, linestyle='--')
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(10, 4))
    for m in range(M):
        for i in range(n_params[m]):
            ax.plot(dates[update_dates[0]+1:], hist_log_pi[m][update_dates[0]:].exp(), linewidth=0.3, c=colors[m])
    ax.vlines(x=dates[update_dates], ymin=0, ymax=1, label='update', colors='black', linestyle='--', linewidth=0.5)
    #ax.plot(dates[update_dates[0]:], torch.log(S[update_dates[0]:] / S[update_dates[0]-1:-1]) + 0.25, linewidth=0.5, linestyle='--')
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(10, 4))
    for m in range(M):
        ax.plot(dates[update_dates[0]:], hist_log_prior[m][update_dates[0]:].exp(), linewidth=1.5, c=colors[m], label=f'{model_classes[m].__class__.__name__}')
    #ax.vlines(x=dates[update_dates], ymin=0, ymax=2., label='update', colors='black', linestyle='--', linewidth=0.5)
    ax.plot(dates[update_dates[0]:], 3 * torch.log(S[update_dates[0]:] / S[update_dates[0]-1:-1]) + 1.5, linewidth=0.5, linestyle='--', label='returns')
    fig.legend()
    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
