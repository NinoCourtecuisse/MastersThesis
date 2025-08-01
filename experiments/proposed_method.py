import argparse

import torch
from torch import distributions as D
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from src.utils.data import load_data
from src.utils.priors import IndependentPrior, CevPrior
from src.utils.distributions import ScaledBeta

from src.models import ModelPool
from src.models import Bs, Cev, Nig

"""

Usage:
    ./run.sh experiments/proposed_method.py
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()

def main(args):
    torch.manual_seed(args.seed)

    # === Load market data ===
    path = 'data/spx_spot.csv'
    dates, S = load_data(path, start='2006-01-01', end='2024-01-01')
    dt = torch.tensor(1 / 252)
    T = len(S) - 1

    # === Hyper-parameters ===
    N = 100                     # Number of particles per model class
    batch_size = 50
    first_update = 100          # Wait to have enough data to do the first update
    update_freq = 20            # Frequency (in days) at which we update the particles
    window = 252                # Only use "recent" past data to update the particles
    lrs = [1e-4, 1e-4, 1e-3]    # Learning rate to update the particles for each model class

    update_dates = torch.arange(start=first_update, end=T+1, step=update_freq)
    n_updates = len(update_dates)

    # === Instantiate the model classes ===
    bs_prior = IndependentPrior([
        D.Normal(0., 0.05),
        ScaledBeta(2.0, 2.0, low=0.01, high=1.0)
    ])
    bs_model = Bs(dt, bs_prior)

    cev_prior = CevPrior(
        mu_dist = D.Normal(0., 0.05),
        beta_dist = ScaledBeta(5., 5., low=torch.tensor(0.5), high=torch.tensor(2.0)),
        v = 0.2, S=1000
    )
    cev_model = Cev(dt, cev_prior)

    nig_prior = IndependentPrior([
        D.Normal(0., 0.05),
        ScaledBeta(2.0, 2.0, low=0.01, high=1.0),
        D.Normal(-2.0, 2.0),
        ScaledBeta(1.5, 5.0, low=1e-6, high=0.1)
    ])
    nig_model = Nig(dt, nig_prior)

    # === Proposed method ===
    model_classes = [bs_model, cev_model, nig_model]
    M = len(model_classes)

    # Initial prior on the model class: Uniform
    init_model_classes_log_prior = torch.log(torch.ones(size=(M,)) / M)

    # Initial particles: Sampled from the prior and mapped to unconstrained space
    init_particles = [
        model.transform.inv(
            model.prior.sample(n_samples = N)
        ) for model in model_classes
    ]
    model_pool = ModelPool(
        model_classes=model_classes,
        init_log_prior=init_model_classes_log_prior,
        init_particles=init_particles
    )
    dims = [len(p[0, :]) for p in model_pool.particles]  # Dimension of the parameter vector of all model classes m=1, ..., M

    # Save for plot: 
    #   History of all particles at all times
    #   Estimate Nothing weights for all particles at all times
    #   Log-likelihood of all particles at all times
    #   Prior on model classes at all times
    hist_particles = [torch.zeros(size=(n_updates+2, N, d)) for d in dims]
    hist_log_pi = [torch.zeros(size=(T, N)) for m in range(M)]
    hist_ll = [torch.zeros(size=(T, N)) for m in range(M)]
    hist_log_prior = [torch.zeros(size=(T+1,)) for m in range(M)]

    for m in range(M):
        hist_particles[m][0, :, :] = model_classes[m].transform.to(init_particles[m])
    for m in range(M):
        hist_log_prior[m][0] = init_model_classes_log_prior[m]

    for i in range(n_updates + 1):              #TODO: Change i to k
        # Loop of t_k for k=1, ..., K
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

        # Step 1: For t=t_{k-1}+1, ..., t_k
        #   Follow "Estimate Nothing"
        log_pi, log_iw, log_post, ll = model_pool.estimate_nothing(
            data=S[t_k_minus_1:t_k+1],
            temperature=1.
        )

        # Step 2: Update the particles
        #   2.1: Update the prior on model classes
        if i > 0:
            beta = 0.8
            model_classes_log_prior = beta * log_post[:, -1] + (1 - beta) * init_model_classes_log_prior
            model_classes_log_prior -= torch.logsumexp(model_classes_log_prior, dim=0)
            model_pool.log_prior = model_classes_log_prior

        #   2.2 and 2.3: Resample and move 
        dataset = TensorDataset(S[max(t_k+1-window, 0):t_k+1])      #TODO: Useless to use Dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        n_batch = len(dataloader)
        batch_weights = torch.ones(size=(n_batch,)) / n_batch       #TODO: Delete weights

        model_pool.update_particles(
            dataloader,
            log_iw,
            n_grad_steps=100,
            lrs=lrs,
            batch_weights=batch_weights
        )
        new_particles = model_pool.particles

        # Save
        for m in range(M):
            hist_particles[m][i+1, :, :] = model_classes[m].transform.to(new_particles[m])
            hist_log_pi[m][t_k_minus_1:t_k] = log_pi[m]
            hist_ll[m][t_k_minus_1:t_k] = ll[m]
            hist_log_prior[m][t_k_minus_1+1:t_k] = log_post[m, :-1]
            hist_log_prior[m][t_k] = model_pool.log_prior[m]

    # === Plot results ===
    colors = ['blue', 'red', 'green']
    params_names = [
        ['\\mu', '\\sigma'],
        ['\\mu', '\\delta', '\\beta'],
        ['\\mu', '\\sigma', '\\xi', '\\eta']
    ]

    # Plot all particles at all times for all model classes
    for m in range(M):
        fig, axes = plt.subplots(dims[m], 1, figsize=(10, 4), sharex=True)
        param_str = ", ".join(params_names[m])
        title = (
            f"Set of parameters considered for model class {model_classes[m].__class__.__name__}\n"
            f"${param_str}$ from top to bottom."
        )
        axes[0].set_title(title)
        for i in range(dims[m]):
            ax = axes[i]
            for n in range(N):
                ax.scatter(dates[update_dates], hist_particles[m][1:-1, n, i], marker='+', s=10.0)
            ax.grid()
        fig.tight_layout()

    # Plot the unnormalized "Estimate Nothing" weights for all particles at all times
    fig, ax = plt.subplots(figsize=(10, 4))
    for m in range(M):
        for i in range(dims[m]):
            ax.plot(dates[update_dates[0]+1:], hist_ll[m][update_dates[0]:], linewidth=0.3, c=colors[m])
    ax.vlines(x=dates[update_dates], ymin=0, ymax=1, label='update', colors='black', linestyle='--', linewidth=0.5)
    fig.tight_layout()

    # Plot the "Estimate Nothing" weights for all particles at all times
    fig, ax = plt.subplots(figsize=(10, 4))
    for m in range(M):
        for i in range(dims[m]):
            ax.plot(dates[update_dates[0]+1:], hist_log_pi[m][update_dates[0]:].exp(), linewidth=0.3, c=colors[m])
    ax.vlines(x=dates[update_dates], ymin=0, ymax=1, label='update', colors='black', linestyle='--', linewidth=0.5)
    fig.tight_layout()

    # Plot the priors on the model classes at all times
    fig, ax = plt.subplots(figsize=(10, 4))
    for m in range(M):
        ax.plot(dates[update_dates[0]:], hist_log_prior[m][update_dates[0]:].exp(),
                linewidth=1.5, c=colors[m], label=f'{model_classes[m].__class__.__name__}')
    ax.plot(dates[update_dates[0]:], 3 * torch.log(S[update_dates[0]:] / S[update_dates[0]-1:-1]) + 1.5, linewidth=0.5, linestyle='--', label='returns')
    fig.legend()
    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
