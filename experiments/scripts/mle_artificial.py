import argparse

import torch
from torch import distributions as D
import matplotlib.pyplot as plt

from utils.priors import IndependentPrior, CevPrior, NigPrior
from utils.distributions import ScaledBeta

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['bs', 'cev', 'nig', 'sv'])
    parser.add_argument('--save', type=str, help='Path to save the plot.')
    parser.add_argument('--seed', type=int, default=0)
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
            prior = CevPrior(
                mu_dist = D.Uniform(-0.5, 0.5),
                beta_dist = ScaledBeta(5., 5., low=torch.tensor(0.5), high=torch.tensor(2.0)),
                v = 0.2, S=100
            )
        case 'nig':
            from models import Nig as Model
            params_true = torch.tensor([
                [0.0, 0.2, -0.05, 0.05] # mu, sigma, gamma_1, gamma_2
            ])
            prior = NigPrior(
                mu_dist=D.Normal(0., 0.1),
                sigma_dist=ScaledBeta(2.0, 2.0, low=0.01, high=1.5),
                gamma1_dist=ScaledBeta(2.0, 2.0, low=-0.2, high=0.2)
            )
        case 'sv':
            from models import Sv as Model
            params_true = torch.tensor([
                [0.0, 0.1, 4.0, 0.9, -0.5] # mu, sigma_y, sigma_h, phi, rho
            ])
            prior = IndependentPrior([
                D.Uniform(low=-0.2, high=0.2),
                D.Uniform(low=1e-4, high=1.0),
                D.Uniform(low=0.1, high=10.0),
                D.Uniform(low=-0.99, high=0.99),
                D.Uniform(low=-0.99, high=0.0)
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
    
    fig1 = plt.figure(figsize=(8, 5))
    plt.plot(torch.linspace(0, T, len(S)), S)
    fig1.tight_layout()
    if args.save:
        fig1.savefig(fname=f'{args.save}/{args.model}_paths.png', bbox_inches='tight')

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
        print(f'Path {i+1} / {n_paths}')
        data = S[:, i]

        if args.model == 'sv':
            model.build_objective(data)
        else:
            if args.model == 'cev': lr = 1.0
            if args.model == 'nig': lr = 0.1

        params = prior.sample()
        opt_params = model.transform.inv(params).requires_grad_(True)

        optimizer = torch.optim.Adam([opt_params], lr=lr)
        for j in range(max_it):
            optimizer.zero_grad()
            loss = -model.ll(opt_params, data)
            loss.backward()

            grad_norm = torch.norm(opt_params.grad)
            #print(f'Epoch {j}, Loss: {loss.item():.3f}, Grad Norm: {grad_norm.item():.3f}')

            if grad_norm < grad_norm_threshold:     # Stop the optimization
                stats['n_it'][i] = j
                break
            if j == max_it - 1:
                stats['n_it'][i] = j
                stats['no_convergence'] += 1
                print('Maximum iteration reached.')
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

    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
