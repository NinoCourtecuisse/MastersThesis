import torch
from torch import distributions as D
import matplotlib.pyplot as plt

from src.utils.distributions import NormalInverseGaussian

def main():
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

    plt.show()

if __name__ == '__main__':
    main()
