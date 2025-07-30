import argparse

import torch
import matplotlib.pyplot as plt

from torch import distributions as D
from utils.distributions import NormalInverseGaussian

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save', type=str, help='Path to save the plot.')
    return parser.parse_args()

def main(args):
    torch.manual_seed(args.seed)
    params = {
        'mu': torch.tensor([0., 0., 0.]),
        'sigma': torch.tensor([1., 1., 1.]),
        'xi': torch.tensor([-5., -5., 5.]),
        'eta': torch.tensor([0.05, 0.2, 0.05])
    }

    d = NormalInverseGaussian.from_moments(
        mu=params['mu'].unsqueeze(1),
        sigma=params['sigma'].unsqueeze(1),
        xi=params['xi'].unsqueeze(1),
        eta=params['eta'].unsqueeze(1)
    )

    eval = torch.linspace(-5., 5., 100)
    nig_pdf = d.log_prob(eval).exp()
    gauss_pdf = D.Normal(0., 1.).log_prob(eval).exp()

    fig1 = plt.figure()
    plt.plot(eval, gauss_pdf, label='Gaussian', linestyle='--')
    n_params = params['mu'].shape[0]
    for i in range(n_params):
        plt.plot(eval, nig_pdf[i, :], label=fr'$\xi$={params["xi"][i]:.2f}, $\eta$={params["eta"][i]:.2f}')
    fig1.legend()
    fig1.tight_layout()

    if args.save:
        fig1.savefig(fname=f'{args.save}/nig_distribution.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
