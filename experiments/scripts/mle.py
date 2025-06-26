import argparse

import torch
from torch import distributions as D

from utils.priors import IndependentPrior

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['bs', 'cev', 'nig', 'sv'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()

def main(args):
    match args.model:
        case 'bs':
            from models import Bs as Model
            params_true = torch.tensor([
                [0.1, 0.2] # mu, sigma
            ])
            prior = IndependentPrior([
                D.Uniform(-0.5, 0.5),
                D.Uniform(1e-4, 1.0)
            ])
        case 'cev':
            from models import Cev as Model
            params_true = torch.tensor([
                [0.1, 2.0, 1.0] # mu, delta, beta
            ])
            prior = IndependentPrior([
                D.Uniform(-0.5, 0.5),
                D.Uniform(0.01, 5.),
                D.Uniform(1e-4, 2.5)
            ])
        case 'nig':
            from models import Nig as Model
            params_true = torch.tensor([
                [0.1, 0.2, -1.0, 0.01] # mu, sigma, xi, eta
            ])
            prior = IndependentPrior([
                D.Uniform(-0.5, 0.5),
                D.Uniform(1e-4, 1.0),
                D.Uniform(-5., 5.),
                D.Uniform(1e-4, 0.5)
            ])
        case 'sv':
            from models import Sv as Model
            params_true = torch.tensor([
                [0.01, 0.25, 0.9, -0.5, 0.001] # sigma_y, sigma_h, phi, rho, mu
            ])
            prior = IndependentPrior([
                D.Uniform(low=1e-4, high=1.0),
                D.Uniform(low=1e-4, high=1.0),
                D.Uniform(low=-0.99, high=0.99),
                D.Uniform(low=-0.99, high=0.0),
                D.Uniform(low=0.0001, high=0.01)
            ])
        case _:
            return NotImplementedError(f'Model {args.model} is currently not implemented.')

    torch.manual_seed(args.seed)

    ######## Instantiate the model ########
    dt = 1 / 252
    model = Model(dt, prior)

    ######## Generate paths ########
    T = torch.tensor(1.0)
    s0 = torch.tensor(100.0)

    if args.model == 'sv':
        x, h = model.simulate(params_true, s0, T, M=100)
        S = torch.exp(x)
    else:
        S = model.simulate(params_true, s0, T, M=1000)

    ######## Maximize the log-likelihood for each path ########
    n_paths = S.shape[1]
    n_params = len(params_true.squeeze())
    stats = {
        'mle': torch.zeros(size=(n_paths, n_params)),
        'max_value': torch.zeros(size=(n_paths,)),
        'n_it': torch.zeros(size=(n_paths,)),
        'no_convergence': 0,
    }
    grad_norm_threshold = 0.1
    lr = 0.5
    max_it = 500
    for i in range(n_paths):
        if args.verbose: print(f'Path {i+1} / {n_paths}')
        data = S[:, i]

        if args.model == 'sv':
            model.build_objective(data)
        else:
            if args.model == 'cev': lr = 1.0

        params = prior.sample()
        opt_params = model.transform.inv(params).requires_grad_(True)

        optimizer = torch.optim.Adam([opt_params], lr=lr)
        for j in range(max_it):
            optimizer.zero_grad()
            loss = -model.ll(opt_params, data)
            loss.backward()

            grad_norm = torch.norm(opt_params.grad)
            #if args.verbose: print(f'Epoch {j}, Loss: {loss.item():.3f}, Grad Norm: {grad_norm.item():.3f}')

            if grad_norm < grad_norm_threshold:     # Stop the optimization
                stats['n_it'][i] = j
                break
            if j == max_it - 1:
                stats['n_it'][i] = j
                stats['no_convergence'] += 1
                if args.verbose: print('Maximum iteration reached.')
            optimizer.step()

        final_params = model.transform.to(opt_params)
        stats['mle'][i, :] = final_params[0, :]

    ######## Compute some statistics ########
    mean = torch.mean(stats['mle'], dim=0)
    std = torch.std(stats['mle'], dim=0)
    avg_it = torch.mean(stats['n_it'])
    n_no_conv = stats['no_convergence']

    print(f'Mean: {mean}')
    print(f'Std: {std}')
    print(f'Average number of it: {avg_it}')
    print(f'No convergence: {n_no_conv}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
