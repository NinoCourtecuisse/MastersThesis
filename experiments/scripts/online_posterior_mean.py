import argparse

import torch
from torch import distributions as D
import matplotlib.pyplot as plt

from utils.data import load_data, batch_data
from utils.priors import IndependentPrior, CevPrior
from utils.distributions import ScaledBeta
from utils.posterior import summary

from inference.ibis import backtest
from inference import KernelSGLD

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['bs', 'cev'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--save', type=str, help='Path to save the plot.')
    return parser.parse_args()

def main(args):
    torch.manual_seed(args.seed)

    ######## Load data ########
    path = 'data/spx_spot.csv'
    dates, s = load_data(path)

    batch_size = 15
    s_batch = batch_data(s, batch_size=batch_size)
    n_batch = s_batch.shape[0]
    if args.verbose: print(f"Batched dataset shape: {s_batch.shape}")

    ######## Hyper parameters ########
    dt = torch.tensor(1 / 252)
    ESS_rmin = 0.5
    window = 100
    kernel = KernelSGLD(n_steps=50, lr=1e-2, lr_min=1e-4, gamma=0.51)
    n_particles = 50
    verbose = args.verbose

    ########## Instantiate the model ##########
    match args.model:
        case 'bs':
            from models import Bs as Model
            params_name = ['mu', 'sigma']
            prior = IndependentPrior([
                D.Normal(0.0, 0.1),     # mu
                D.Uniform(0.01, 1.0)    # sigma
            ])
        case 'cev':
            from models import Cev as Model
            params_name = ['mu', 'delta', 'beta']
            prior = CevPrior(
                mu_dist=D.Normal(0., 0.1),
                beta_dist=ScaledBeta(5., 5., low=torch.tensor(0.5), high=torch.tensor(2.0))
            )

    model = Model(dt, prior)
    hist = backtest(model, kernel, s_batch, s, n_particles, ESS_rmin, window, verbose)

    ########## Compute the online posterior mean and std dev ##########
    particles_hist = hist['particles']
    weights_hist = hist['weights']

    n_params = len(particles_hist[0][0])
    mean = torch.zeros((n_batch, n_params))
    ci_low = torch.zeros((n_batch, n_params))
    ci_high = torch.zeros((n_batch, n_params))
    for t in range(n_batch):
        m, q1, q2 = summary(particles_hist[t], weights_hist[t])
        mean[t, :] = m
        ci_low[t, :] = q1
        ci_high[t, :] = q2

    ########## Plots ##########
    dates_plot = dates[batch_size::batch_size]

    # Online posterior mean, with credible interval
    fig1 = plt.figure(figsize=(10, 4))
    for i in range(n_params):
        plt.subplot(n_params+1, 1, i+1)
        plt.plot(dates_plot, mean[:, i], label=rf'$\{params_name[i]}$', linewidth=0.8)
        plt.fill_between(dates_plot, ci_low[:, i], ci_high[:, i], alpha=0.3)
        plt.legend()
        plt.grid()
    fig1.tight_layout()

    # ESS at each time
    fig2 = plt.figure()
    plt.plot(dates_plot, torch.tensor(hist['ess']) / n_particles, linestyle='--', color='gray', marker='+', linewidth=0.1)
    plt.hlines(y=ESS_rmin, xmin=dates_plot[0], xmax=dates_plot[-1], color='black', label='ESS rmin')
    plt.ylabel(r'$ESS / \# particles$')

    if args.model == 'cev':
        plt.figure()
        mean_vol = torch.zeros((n_batch, ))
        for t in range(n_batch):
            local_vol = particles_hist[t][:, 1] * s[(t+1) * batch_size]**(particles_hist[t][:, 2]/2 - 1)
            mean_vol[t] = (weights_hist[t] * local_vol).sum()
        plt.plot(dates_plot, mean_vol, linewidth=0.8)
        plt.grid()

    # Contour of log-posterior and particles at 4 dates through the period
    fig3, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten() 
    batch_idx = [0, 45, 120, 240]

    match args.model:
        case 'bs':
            slice = (0, 1)
            x = torch.linspace(-0.3, 0.35, 100)  # mu
            y = torch.linspace(1e-3, 1.0, 100)  # sigma
            x_grid, y_grid = torch.meshgrid([x, y], indexing='ij')
            pairs = torch.column_stack((x_grid.ravel(), y_grid.ravel()))
        case 'cev':
            slice = (1, 2)
            mu_eval = torch.tensor(0.0)
            x = torch.linspace(1e-3, 100, 100)  # delta
            y = torch.linspace(0.5, 2.0, 100)   # beta
            x_grid, y_grid = torch.meshgrid([x, y], indexing='ij')
            pairs = torch.column_stack((mu_eval*torch.ones_like(x_grid.ravel()), x_grid.ravel(), y_grid.ravel()))

    transformed_pairs = model.transform.inv(pairs)
    for i in range(len(batch_idx)):
        ax = axes[i]
        idx = batch_idx[i]
        t = (idx + 1) * batch_size
        start = max([t-window, 0])
        end = t
        lpost = model.lpost(transformed_pairs, s[start:end]).view(100, 100)

        lpost_max = lpost.max()
        levels = torch.linspace(0.9 * lpost_max, lpost_max, 50)
        fine_levels = torch.linspace(0.99 * lpost_max, lpost_max, 1)
        contour = ax.contourf(x_grid, y_grid, lpost, levels=levels, extend='min')
        fig3.colorbar(contour, ax=ax)
        ax.contour(x_grid, y_grid, lpost, levels=fine_levels, linewidths=0.3)

        ax.scatter(hist['particles'][idx][:, slice[0]], hist['particles'][idx][:, slice[1]], label='previous particles', marker='+', c='red')
        ax.scatter(hist['particles'][idx+1][:, slice[0]], hist['particles'][idx+1][:, slice[1]], label='particles after IBIS', marker='+', c='green')
        ax.set_title(f'Date {dates[t]}')
        ax.set_xlabel(params_name[slice[0]])
        ax.set_ylabel(params_name[slice[1]])
        ax.legend()
    fig3.tight_layout()

    if args.save:
        fig1.savefig(fname=f'{args.save}/online_mean_{args.model}.png', bbox_inches='tight')
        fig3.savefig(fname=f'{args.save}/lpost_{args.model}.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)