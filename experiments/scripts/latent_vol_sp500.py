import argparse

import torch
from torch import distributions as D
from torch.optim import Adam
import matplotlib.pyplot as plt

from utils.data import load_data
from utils.priors import IndependentPrior

from models import Sv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()

def main(args):
    ######## Load data ########
    path = 'data/spx_spot.csv'
    dates, S = load_data(path)

    ######## Hyper parameters ########
    dt = 1 / 252

    ######## Instantiate the model ########
    prior = IndependentPrior([
        D.LogNormal(0., 1.),
        D.LogNormal(0., 1.),
        D.Uniform(-1., 1.),
        D.Uniform(-1., 1.),
        D.Normal(0., 1.)
    ])
    sv_model = Sv(dt, prior)

    sv_model.build_objective(data=S)
    params_init = torch.tensor([[0.01, 0.25, 0.95, -0.7, 0.0]])

    params = sv_model.transform.inv(params_init).requires_grad_(True)
    optimizer = Adam([params], lr=0.01)
    n_iter = 300
    for i in range(n_iter):
        optimizer.zero_grad()
        loss = - sv_model.ll(params, data=S)
        loss.backward()

        if args.verbose:
            grad_norm = torch.norm(params.grad)
            print(f"Iteration {i} / {n_iter}: grad norm={grad_norm.item():.3f}")
        optimizer.step()

    final_params = sv_model.transform.to(params.detach())
    sigma_y = final_params[0, 0]

    h, std = sv_model.get_latent(with_std=True)
    h_upper = h + 2 * std
    h_lower = h - 2 * std

    vol = sigma_y * torch.exp(h / 2) * torch.tensor(252).sqrt()
    vol_upper = sigma_y * torch.exp(h_upper / 2) * torch.tensor(252).sqrt()
    vol_lower = sigma_y * torch.exp(h_lower / 2) * torch.tensor(252).sqrt()

    plt.figure(figsize=(10, 3))
    plt.plot(dates[:-1], vol, c='black', linewidth=0.8)
    plt.fill_between(dates[:-1], vol_lower, vol_upper, color='grey', alpha=0.3, label=r'$\pm 2$ std')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
