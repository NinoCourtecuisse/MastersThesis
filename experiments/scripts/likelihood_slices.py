import argparse

import torch
from torch import distributions as D
from torch.optim import Adam
import matplotlib.pyplot as plt
import math

from utils.priors import IndependentPrior, NigPrior
from utils.distributions import ScaledBeta

import torch
import matplotlib.pyplot as plt
import itertools

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['nig', 'sv'])
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()

def main(args):
    torch.manual_seed(args.seed)

    match args.model:
        case 'nig':
            from models import Nig as Model
            prior = NigPrior(
                mu_dist=D.Normal(0., 0.1),
                sigma_dist=D.LogNormal(math.log(0.2), 1.0),
                theta_eta=-math.log(0.01) / 0.1,
                theta_xi=-math.log(0.01) / 5
            )
            params_true = torch.tensor([
                [0.0, 0.2, -1.0, 0.01]  # mu, sigma, xi, eta
            ])
            param_names = ['mu', 'sigma', 'xi', 'eta']
            param_values = {
                'mu': torch.linspace(-1.0, 1.0, 100),
                'sigma': torch.linspace(1e-3, 1.5, 100),
                'xi': torch.linspace(-5.0, 5.0, 100),
                'eta': torch.linspace(1e-4, 0.1, 100)
            }
            fixed_values = {
                'mu': params_true[0, 0], 'sigma': params_true[0, 1],
                'xi': params_true[0, 2], 'eta': params_true[0, 3]
            }
        case 'sv':
            from models import Sv as Model
            prior = IndependentPrior([
                D.Normal(0., 0.1),
                D.LogNormal(math.log(0.2), 1.),
                D.LogNormal(1.5, 0.5),
                ScaledBeta(2.0, 2.0, -1.0, 1.0),
                ScaledBeta(2.0, 2.0, -1.0, 1.0)
            ])
            params_true = torch.tensor([
                [0.0, 0.1, 4.0, 0.9, -0.5] # mu, sigma_y, sigma_h, phi, rho
            ])
            param_names = ['mu', 'sigma_y', 'sigma_h', 'phi', 'rho']
            param_values = {
                'mu': torch.linspace(-0.5, 0.5, 30),
                'sigma_y': torch.linspace(1e-3, 0.5, 30),
                'sigma_h': torch.linspace(1.0, 6.0, 30),
                'phi': torch.linspace(0.5, 0.99, 30),
                'rho': torch.linspace(-0.99, 0.0, 30)
            }
            fixed_values = {
                'mu': params_true[0, 0], 'sigma_y': params_true[0, 1], 
                'sigma_h': params_true[0, 2], 'phi': params_true[0, 3], 'rho': params_true[0, 4]
            }

    dt = 1 / 252
    model = Model(dt, prior)

    T = 1000 / 252
    s0 = torch.tensor(100.0)

    if args.model == 'sv':
        x, h = model.simulate(params_true, s0, T, M=1)
        S = torch.exp(x).squeeze()
        model.build_objective(data=S)
    else:
        S = model.simulate(params_true, s0, T, M=1).squeeze()

    combinations = list(itertools.combinations(param_names, 2))
    for p1, p2 in combinations:
        grid1, grid2 = torch.meshgrid(param_values[p1], param_values[p2], indexing='ij')
        grid1 = grid1.flatten()
        grid2 = grid2.flatten()

        n = grid1.numel()
        n_params = len(param_names)
        theta = torch.zeros((n, n_params))
        for i, name in enumerate(param_names):
            if name == p1:
                theta[:, i] = grid1.reshape(-1)
            elif name == p2:
                theta[:, i] = grid2.reshape(-1)
            else:
                theta[:, i] = fixed_values[name]

        theta_u = model.transform.inv(theta)
        ll = model.ll(theta_u, data=S)
        ll = ll.reshape(grid1.shape)

        # Plot
        plt.figure(figsize=(6, 5))
        ll_max = ll.max()
        levels = torch.linspace(0.9 * ll_max, ll_max, 10)
        fine_levels = torch.linspace(0.999 * ll_max, ll_max, 2)
        contour = plt.tricontourf(grid1, grid2, ll, levels=levels, extend='min')
        plt.colorbar(contour)
        plt.tricontour(grid1, grid2, ll, levels=fine_levels)

        plt.scatter(fixed_values[p1], fixed_values[p2], label='true', marker='+', c='red')
        plt.xlabel(p1)
        plt.ylabel(p2)
        fixed_str = ', '.join(f'{k}={v:.2g}' for k, v in fixed_values.items() if k not in (p1, p2))
        plt.title(f'{p1} vs {p2} | {fixed_str}')
        plt.legend()
        plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
