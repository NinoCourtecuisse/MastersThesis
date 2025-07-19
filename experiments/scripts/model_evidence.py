import argparse

import torch
from torch import distributions as D
import matplotlib.pyplot as plt
import math

from utils.data import load_data
from utils.priors import IndependentPrior, CevPrior
from utils.distributions import ScaledBeta

from models import Bs, Cev, Nig, Sv, Sabr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()

def compute_log_evidence(model, data, start, stop, step, window, mc_samples=10000, verbose=False):
    model_levidence = []  # model log evidence

    for t in range(start, stop, step):
        if verbose: print(f'Day {t} / {stop-1}')

        current_data = data[t-window:t]
        with torch.no_grad():
            prior_samples = model.prior.sample(n_samples=mc_samples)
            theta = model.transform.inv(prior_samples)
            model_levidence.append(torch.logsumexp(model.ll(theta, current_data), dim=0) - math.log(mc_samples))

    model_levidence = torch.tensor(model_levidence).unsqueeze(1)
    return model_levidence

def main(args):
    torch.manual_seed(args.seed)

    ######## Load data ########
    path = 'data/spx_spot.csv'
    dates, S = load_data(path, start='2006-01-01', end='2012-12-31')

    ######## Hyper parameters ########
    dt = 1 / 252
    window = 252
    start = window
    stop = len(S) + 1
    step = 20

    ######## Instantiate the model ########
    bs_prior = IndependentPrior([
        D.Normal(0., 1.0),
        ScaledBeta(2.0, 2.0, low=0.01, high=1.0)
    ])
    bs_model = Bs(dt, bs_prior)

    cev_prior = CevPrior(
        mu_dist = D.Normal(0., 1.),
        beta_dist = ScaledBeta(5., 5., low=torch.tensor(0.5), high=torch.tensor(2.0)),
        v = 0.2, S=1000
    )
    cev_model = Cev(dt, cev_prior)

    nig_prior = IndependentPrior([
        D.Normal(0., 1.0),
        ScaledBeta(2.0, 2.0, low=0.01, high=1.0),
        D.Normal(-2.0, 2.0),
        ScaledBeta(1.5, 5.0, low=1e-6, high=0.1)
    ])
    nig_model = Nig(dt, nig_prior)

    sv_prior = IndependentPrior([
        D.Normal(0., 1.0),
        D.LogNormal(math.log(1.0) - 0.5 * 1.0**2, 1.0),
        D.LogNormal(math.log(5.0) - 0.5 * 0.8**2, 0.8),
        ScaledBeta(3.0, 1.5, -1.0, 1.0),
        ScaledBeta(1.5, 3.0, -1.0, 1.0)
    ])
    sv_model = Sv(dt, sv_prior)

    models = [bs_model, cev_model, nig_model, sv_model]
    n_models = len(models)

    ######## Compute the log evidence of each model ########
    levidences = []
    for i in range(n_models):
        model = models[i]
        model_levidence = compute_log_evidence(model, data=S, start=start, stop=stop, step=step,
                                               window=window, mc_samples=5000, verbose=True)
        levidences.append(model_levidence)

    # Compute model posterior, with uniform prior
    levidences = torch.cat(levidences, dim=1)
    lnorm_constant = torch.logsumexp(levidences, dim=1, keepdim=True)   # Log normalizing constant
    lposteriors = levidences - lnorm_constant

    ######## Plot ########
    dates_plot = dates[start-1:stop:step]

    fig1, ax = plt.subplots(figsize=(10, 4))
    for i in range(n_models):
        ax.plot(dates_plot, levidences[:, i], linewidth=0.8, label=f'{models[i].__class__.__name__}')
    ax.legend()
    ax.grid()
    ax.set_title(label='Models log evidences')

    fig2, ax = plt.subplots(figsize=(10, 4))
    for i in range(n_models):
        ax.plot(dates_plot, lposteriors[:, i], linewidth=0.8, label=f'{models[i].__class__.__name__}')
    ax.legend()
    ax.grid()
    ax.set_title(label='Models log posteriors')

    fig3, ax = plt.subplots(figsize=(10, 4))
    for i in range(n_models):
        ax.plot(dates_plot, lposteriors[:, i].exp(), linewidth=0.8, label=f'{models[i].__class__.__name__}')
    ax.legend()
    ax.grid()
    ax.set_title(label='Models posteriors')

    plt.show()
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)