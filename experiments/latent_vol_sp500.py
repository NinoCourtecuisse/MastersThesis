import argparse

import torch
from torch import distributions as D
from torch.optim import Adam
import matplotlib.pyplot as plt

from src.utils.data import load_data
from src.utils.priors import IndependentPrior

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['sv', 'sabr'], required=True)
    return parser.parse_args()

def main(args):
    ######## Load data ########
    path = 'data/spx_spot.csv'
    dates, S = load_data(path)

    ######## Hyper parameters ########
    dt = 1 / 252

    ######## Instantiate the model ########
    match args.model:
        case 'sv':
            from src.models import Sv as Model
            prior = IndependentPrior([
                D.Uniform(-0.1, 0.1),   # mu, sigma_y, sigma_h, phi, rho
                D.LogNormal(0., 1.),
                D.LogNormal(0., 1.),
                D.Uniform(-1., 1.),
                D.Uniform(-1., 1.)
            ])
            params_init = torch.tensor([[-0.08, 1.0, 1.0, -0.5, -0.8]])
        case 'sabr':
            from src.models import Sabr as Model
            prior = IndependentPrior([
                D.Uniform(-1e-2, 1e-2),       # mu, beta, sigma, rho
                D.LogNormal(0., 1.),
                D.LogNormal(0., 1.),
                D.Uniform(-1., 1.)
            ])
            params_init = torch.tensor([[0.1, 1.0, 0.2, 0.0]])
        case _:
            raise NotImplementedError(f"Model {args.model} is not implemented.")

    model = Model(dt, prior)
    model.build_objective(data=S)

    ######## MLE and retrieve the latent ########
    params = model.transform.inv(params_init).requires_grad_(True)
    optimizer = Adam([params], lr=0.1)
    #optimizer = torch.optim.SGD([params], lr=2*1e-4)
    n_iter = 300
    for i in range(n_iter):
        optimizer.zero_grad()
        loss = - model.ll(params, data=S)
        loss.backward()

        grad_norm = torch.norm(params.grad)
        print(f"Iteration {i} / {n_iter}: grad norm={grad_norm.item():.3f}")
        optimizer.step()

    final_params = model.transform.to(params.detach())
    print(f'Final params: {final_params}')
    h, std = model.get_latent(with_std=True)
    h_upper = h + 2 * std
    h_lower = h - 2 * std

    match args.model:
        case 'sv':
            sigma_y = final_params[0, 1]
            var = sigma_y**2 * torch.exp(h)
            var_upper = sigma_y**2 * torch.exp(h_upper)
            var_lower = sigma_y**2 * torch.exp(h_lower)
        case 'sabr':
            beta = final_params[0, 1]
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
