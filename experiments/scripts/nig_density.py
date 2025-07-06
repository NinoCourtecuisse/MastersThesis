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
        'gamma_1': torch.tensor([-2., -2., 2.]),
        'gamma_2': torch.tensor([10., 20., 10.])
    }

    d = NormalInverseGaussian.from_moments(
        mu=params['mu'].unsqueeze(1),
        sigma=params['sigma'].unsqueeze(1),
        gamma_1=params['gamma_1'].unsqueeze(1),
        gamma_2=params['gamma_2'].unsqueeze(1)
    )

    eval = torch.linspace(-5., 5., 100)
    nig_pdf = d.log_prob(eval).exp()
    gauss_pdf = D.Normal(0., 1.).log_prob(eval).exp()

    fig1 = plt.figure()
    plt.plot(eval, gauss_pdf, label='Gaussian', linestyle='--')
    n_params = params['mu'].shape[0]
    for i in range(n_params):
        plt.plot(eval, nig_pdf[i, :], label=fr'$\gamma_1$={params["gamma_1"][i]}, $\gamma_2$={params["gamma_2"][i]}')
    fig1.legend()
    fig1.tight_layout()

    if args.save:
        fig1.savefig(fname=f'{args.save}/nig_distribution.png', bbox_inches='tight')
    plt.show()

    # Sanity check of the reparametrization: sample and compare empirical moments
    d = NormalInverseGaussian.from_moments(
        mu=params['mu'][0],
        sigma=params['sigma'][0],
        gamma_1=params['gamma_1'][0],
        gamma_2=params['gamma_2'][0]
    )
    samples = d.sample(sample_shape=(100000,))
    mean = samples.mean()
    std = samples.std()
    centered = samples - mean
    skewness = (centered**3).mean() / (std**3)
    kurtosis = (centered**4).mean() / (std**4)
    ekurtosis = kurtosis - 3.

    print('Empricial moments: '
          f'Mean={mean:.3f} '
          f'Std dev={std:.3f} '
          f'Skewness={skewness:.3f} '
          f'E. kurtosis={ekurtosis:.3f} '
    )
    print('True moments: '
          f'Mean={params["mu"][0]:.3f} '
          f'Std dev={params["sigma"][0]:.3f} '
          f'Skewness={params["gamma_1"][0]:.3f} '
          f'E. kurtosis={params["gamma_2"][0]:.3f} '
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)
