import argparse

import torch
from torch import distributions as D
import matplotlib.pyplot as plt
import math

from utils.data import load_data, batch_data
from utils.special_functions import sliding_sum
from utils.priors import IndependentPrior, CevPrior, NigPrior
from utils.distributions import ScaledBeta

from inference.ibis import backtest
from inference import KernelSGLD
from models import Bs, Cev, Nig, Sv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--save', type=str, help='Path to save the plot.')
    return parser.parse_args()

def main(args):
    torch.manual_seed(args.seed)

    ######## Load data ########
    path = 'data/spx_spot.csv'
    #dates, s = load_data(path, start='2006-01-03', end='2012-12-31')
    dates, s = load_data(path)

    batch_size = 15
    s_batch = batch_data(s, batch_size=batch_size)
    print(f"Batched dataset shape: {s_batch.shape}")

    ######## Hyper parameters ########
    dt = torch.tensor(1 / 252)
    n_models = 4
    ESS_rmin = 0.5
    window = 100
    kernel = KernelSGLD(n_steps=100, lr=1e-2, lr_min=1e-4, gamma=1.0)
    verbose = args.verbose

    ########## Black-Scholes ##########
    n_particles = 100
    bs_prior = IndependentPrior([
        D.Normal(0.0, 0.1),     # mu
        D.Uniform(0.01, 1.5)    # sigma
    ])
    bs_model = Bs(dt, bs_prior)
    bs_hist = backtest(bs_model, kernel, s_batch, s, n_particles, ESS_rmin, window, verbose)

    ########## CEV ##########
    n_particles = 100
    cev_prior = CevPrior(
        mu_dist=D.Normal(0., 0.1),
        beta_dist=ScaledBeta(5., 5., low=torch.tensor(0.5), high=torch.tensor(2.0))
    )
    cev_model = Cev(dt, cev_prior)
    cev_hist = backtest(cev_model, kernel, s_batch, s, n_particles, ESS_rmin, window, verbose)

    ########## NIG ##########
    n_particles = 100
    nig_prior = NigPrior(
        mu_dist=D.Normal(0., 0.1),
        sigma_dist=D.LogNormal(math.log(0.2), 1.0),
        theta_eta=-math.log(0.01) / 0.1,
        theta_xi=-math.log(0.001) / 5.
    )
    nig_model = Nig(dt, nig_prior)
    nig_hist = backtest(nig_model, kernel, s_batch, s, n_particles, ESS_rmin, window, verbose)

    ########## SV ##########
    #n_particles = 100
    #sv_prior = IndependentPrior([
    #    D.Normal(0.0, 0.1),   # mu, sigma_y, sigma_h, phi, rho
    #    D.Uniform(0., 1.5),
    #    D.Uniform(0., 1.5),
    #    D.Uniform(-1., 1.),
    #    D.Uniform(-1., 1.)
    #])
    #sv_model = Sv(dt, sv_prior)
    #sv_hist = backtest(sv_model, kernel, s_batch, s, n_particles, ESS_rmin, window, verbose)

    # Model prior
    model_prior = torch.ones(size=(n_models,)) / n_models

    tau = 0.01 # temperature
    bs_post = sliding_sum(torch.tensor(bs_hist['ll']) * tau, w=window // batch_size) + model_prior[0].log()
    cev_post = sliding_sum(torch.tensor(cev_hist['ll']) * tau, w=window // batch_size) + model_prior[1].log()
    nig_post = sliding_sum(torch.tensor(nig_hist['ll']) * tau, w=window // batch_size) + model_prior[2].log()

    tmp = torch.stack([bs_post, cev_post, nig_post])
    lnorm_constant = torch.logsumexp(tmp, dim=0)
    bs_lpost = bs_post - lnorm_constant
    cev_lpost = cev_post - lnorm_constant
    nig_lpost = nig_post - lnorm_constant

    dates_plot = dates[batch_size::batch_size]
    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(dates_plot, bs_hist['ll'], label='bs')
    plt.plot(dates_plot, cev_hist['ll'], label='cev')
    plt.plot(dates_plot, nig_hist['ll'], label='nig')
    plt.grid()
    plt.legend()
    plt.title('Model marginal log-likelihood')

    fig2 = plt.figure(figsize=(5, 3))
    plt.plot(dates_plot, bs_lpost, label='bs')
    plt.plot(dates_plot, cev_lpost, label='cev')
    plt.plot(dates_plot, nig_lpost, label='nig')
    plt.grid()
    plt.legend()
    plt.title('Model log posterior')

    fig3 = plt.figure(figsize=(5, 3))
    plt.plot(dates_plot, bs_lpost.exp(), label='bs')
    plt.plot(dates_plot, cev_lpost.exp(), label='cev')
    plt.plot(dates_plot, nig_lpost.exp(), label='nig')
    plt.grid()
    plt.legend()
    plt.title('Model posterior')

    if args.save:
        fig3.savefig(fname=f'{args.save}/models_posterior.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
