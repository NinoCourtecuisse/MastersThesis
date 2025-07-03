import argparse

import torch
from torch import distributions as D
import matplotlib.pyplot as plt

from utils.data import load_data, batch_data
from utils.priors import IndependentPrior, CevPrior
from utils.distributions import ScaledBeta

from models import Cev

def main():
    ######## Load data ########
    path = 'data/spx_spot.csv'
    dates, s = load_data(path)

    dt = torch.tensor(1 / 252)
    prior = CevPrior(
        mu_dist=D.Normal(0., 0.1),
        beta_dist=ScaledBeta(5., 5., low=torch.tensor(0.5), high=torch.tensor(2.0))
    )
    model = Cev(dt, prior)

    ######## Compute and plot the ll ########
    mu_eval = torch.tensor(0.)
    delta_eval = torch.linspace(1., 100., 200)
    beta_eval = torch.linspace(0.5, 2.0, 200)
    delta_grid, beta_grid = torch.meshgrid([delta_eval, beta_eval], indexing='ij')
    mu_grid = mu_eval * torch.ones_like(delta_grid)
    params_eval = torch.column_stack((mu_grid.ravel(), delta_grid.ravel(), beta_grid.ravel()))
    transformed_params = model.transform.inv(params_eval)
    lpost = model.lpost(transformed_params, data=s[:100]).view(200, 200)

    plt.figure(figsize=(5, 3))
    lpost_max = lpost.max()
    levels = torch.linspace(0.9 * lpost_max, lpost_max, 50)
    fine_levels = torch.linspace(0.99 * lpost_max, lpost_max, 1)
    contour = plt.contourf(delta_grid.log(), beta_grid, lpost, levels=levels, extend='min')
    plt.colorbar(contour)
    plt.contour(delta_grid.log(), beta_grid, lpost, levels=fine_levels, linewidths=0.3)

    ######## Plot the approximation line ########
    a = torch.tensor(0.2 * 1000).log()
    b = - torch.tensor(1000).log() / 2
    plt.plot(delta_eval.log(), (delta_eval.log() - a) / b, c='red')

    plt.xlabel(r'$\delta$')
    plt.ylabel(r'$\beta$')
    
    plt.show()

if __name__ == '__main__':
    main()
