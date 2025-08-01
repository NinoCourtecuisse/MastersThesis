import argparse

import torch
from torch import distributions as D
from torch.optim import Adam
import matplotlib.pyplot as plt

from src.utils.data import load_data
from src.utils.priors import IndependentPrior

"""
Maximum Likelihood Estimation of some stochastic volatility models.
Based on a Laplace approximation of the marginal likelihood.

Implemented models: sv and sabr.
Print the final MLE and plot the estimated volatility path
at the MLE.

The Laplace approximation is performed by TMB (https://github.com/kaskr/adcomp)
We call it through our PyTorch wrapper.
Usage:
    ./run.sh experiments/latent_vol_sp500.py --model sv
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['sv', 'sabr'], required=True)
    return parser.parse_args()

def main(args):
    # === Load market data ===
    path = 'data/spx_spot.csv'
    dates, S = load_data(path, start='2006-01-01', end='2024-01-01')
    dt = 1 / 252

    # === Instantiate the model: SV or SABR ===
    # Need to define a prior to map the parameters to an unconstrained space.
    # This prior is only used to define a support. Its actual hyperparameters are ignored.
    match args.model:
        case 'sv':
            from src.models import Sv as Model
            prior = IndependentPrior([
                D.Uniform(-0.1, 0.1),   # mu
                D.LogNormal(0., 1.),    # sigma_y > 0
                D.LogNormal(0., 1.),    # sigma_h > 0
                D.Uniform(-1., 1.),     # -1 < phi < 1
                D.Uniform(-1., 1.)      # -1 < rho < 1
            ])
            params_init = torch.tensor([[-0.08, 1.0, 1.0, -0.5, -0.8]])     # Inital guess
        case 'sabr':
            from src.models import Sabr as Model
            prior = IndependentPrior([
                D.Uniform(-1e-2, 1e-2),     # mu
                D.LogNormal(0., 1.),        # beta > 0
                D.LogNormal(0., 1.),        # sigma > 0
                D.Uniform(-1., 1.)          # -1 < rho < 1
            ])
            params_init = torch.tensor([[0.1, 1.0, 0.2, 0.0]])      # Inital guess
        case _:
            raise NotImplementedError(f"Model {args.model} is not implemented.")

    model = Model(dt, prior)
    model.build_objective(data=S)   # First need to give the data to the model, that's how TMB works.

    # === Compute the MLE ===
    u_params = model.transform.inv(params_init).requires_grad_(True)    # Map initial guess to uconstrained space.
    optimizer = Adam([u_params], lr=0.1)
    n_iter = 300
    for i in range(n_iter):
        optimizer.zero_grad()
        loss = - model.ll(u_params, data=S)
        loss.backward()

        grad_norm = torch.norm(u_params.grad)
        print(f"Iteration {i+1} / {n_iter}: grad norm={grad_norm.item():.3f}")
        optimizer.step()

    # === Print the MLE and Plot the estimate volatility path ===
    final_params = model.transform.to(u_params.detach())    # Map back to original space
    print(f'MLE: {final_params}')

    h, std = model.get_latent(with_std=True)
    h_upper = h + 2 * std
    h_lower = h - 2 * std

    # Reconstructs the volatility from the latent variable h
    match args.model:
        case 'sv':
            sigma_y = final_params[0, 1]

            # SV variance:
            var = sigma_y**2 * torch.exp(h)
            var_upper = sigma_y**2 * torch.exp(h_upper)
            var_lower = sigma_y**2 * torch.exp(h_lower)
        case 'sabr':
            beta = final_params[0, 1]

            # SABR local variance:
            var = model.local_var(S, h, beta)
            var_upper = model.local_var(S, h_upper, beta)
            var_lower = model.local_var(S, h_lower, beta)

    vol = torch.sqrt(var)
    vol_upper = torch.sqrt(var_upper)
    vol_lower = torch.sqrt(var_lower)

    plt.figure(figsize=(10, 3))
    plt.plot(dates[:len(vol)], vol, c='black', linewidth=0.8)
    plt.fill_between(dates[:len(vol)], vol_lower, vol_upper, color='grey', alpha=0.3, label=r'$\pm 2$ std')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
