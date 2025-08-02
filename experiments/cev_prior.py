import torch
from torch import distributions as D
import matplotlib.pyplot as plt

from src.utils.data import load_data
from src.utils.priors import CevPrior
from src.utils.distributions import ScaledBeta
from src.models import Cev

"""
Plot the chosen prior for the CEV model.
Contour plot of the likelihood.

Usage:
    ./run.sh experiments/cev_prior.py
"""

def main():
    # === Load market data ===
    path = 'data/spx_spot.csv'
    dates, s = load_data(path, start='2006-01-01', end='2024-01-01')
    dt = torch.tensor(1 / 252)

    # === Define the prior ===
    prior = CevPrior(
        mu_dist=D.Normal(0., 0.1),
        beta_dist=ScaledBeta(5., 5., low=torch.tensor(0.5), high=torch.tensor(2.0))
    )
    model = Cev(dt, prior)

    # === Compute and plot the log-likelihood ===
    # Generate grid of parameters
    mu_eval = torch.tensor(0.)
    delta_eval = torch.linspace(1., 30, 200)
    beta_eval = torch.linspace(0.5, 2.0, 200)
    delta_grid, beta_grid = torch.meshgrid([delta_eval, beta_eval], indexing='ij')
    mu_grid = mu_eval * torch.ones_like(delta_grid)
    params_eval = torch.column_stack((mu_grid.ravel(), delta_grid.ravel(), beta_grid.ravel()))

    transformed_params = model.transform.inv(params_eval)   # Map to unconstrained space to evaluate the log-likelihood
    ll = model.ll(transformed_params, data=s[:100]).view(200, 200)

    fig1 = plt.figure(figsize=(5, 3))
    ll_max = ll.max()
    levels = torch.linspace(0.9 * ll_max, ll_max, 50)
    fine_levels = torch.linspace(0.99 * ll_max, ll_max, 1)
    contour = plt.contourf(delta_grid.log(), beta_grid, ll, levels=levels, extend='min')
    plt.contour(delta_grid.log(), beta_grid, ll, levels=fine_levels, linewidths=0.3)

    # === Plot the approximated linear relation between log(delta) and beta ===
    a = torch.tensor(0.2 * 1000).log()
    b = - torch.tensor(1000).log() / 2
    plt.plot(delta_eval.log(), (delta_eval.log() - a) / b, c='red', label=r'$a_1 + a_2\beta$')

    plt.xlabel(r'$\log\delta$', fontsize=20)
    plt.ylabel(r'$\beta$', fontsize=20)
    plt.legend(fontsize=20)

    plt.show()

if __name__ == '__main__':
    main()
