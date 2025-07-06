import argparse

import torch
from torch import distributions as D
from torch.optim import Adam
import matplotlib.pyplot as plt

from utils.priors import NigPrior
from utils.distributions import ScaledBeta

from models import Nig

import torch
import matplotlib.pyplot as plt
import itertools

def main():
    torch.manual_seed(0)

    prior = NigPrior(
        mu_dist=D.Normal(0., 0.1),
        sigma_dist=ScaledBeta(2.0, 2.0, low=0.01, high=1.5),
        gamma1_dist=ScaledBeta(2.0, 2.0, low=-0.2, high=0.2)
    )
    dt = 1 / 252
    nig_model = Nig(dt, prior)

    params_true = torch.tensor([
        [0.0, 0.2, -0.05, 0.05]
    ])
    T = 2.0
    s0 = torch.tensor(100.0)
    S = nig_model.simulate(params_true, s0, T, M=1).squeeze()

    param_names = ['mu', 'sigma', 'gamma_1', 'gamma_2']
    param_values = {
        'mu': torch.linspace(-1.0, 1.0, 100),
        'sigma': torch.linspace(0.02, 1.4, 100),
        'gamma_1': torch.linspace(-0.2, 0.2, 100),
        'gamma_2': torch.linspace(1e-3, 0.4, 100)
    }
    fixed_values = {'mu': params_true[0, 0], 'sigma': params_true[0, 1], 'gamma_1': params_true[0, 2], 'gamma_2': params_true[0, 3]}

    combinations = list(itertools.combinations(param_names, 2))
    for p1, p2 in combinations:
        grid1, grid2 = torch.meshgrid(param_values[p1], param_values[p2], indexing='ij')
        if p2 == 'gamma_2':
            if p1 == 'gamma_1':
                mask = grid2 > 5 * grid1**2 / 3
            else:
                mask = grid2 > 5 * fixed_values['gamma_1']**2 / 3
        elif p2 == 'gamma_1':
            mask = fixed_values['gamma_2'] > 5 * grid2**2 / 2
        else:
            mask = torch.ones_like(grid1, dtype=torch.bool)

        grid1 = grid1[mask]
        grid2 = grid2[mask]

        n = grid1.numel()
        theta = torch.zeros((n, 4))
        for i, name in enumerate(param_names):
            if name == p1:
                theta[:, i] = grid1.reshape(-1)
            elif name == p2:
                theta[:, i] = grid2.reshape(-1)
            else:
                theta[:, i] = fixed_values[name]

        theta_u = nig_model.transform.inv(theta)
        ll = nig_model.ll(theta_u, data=S)
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
    main()