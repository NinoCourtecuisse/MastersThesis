import argparse

import torch
from torch import distributions as D
from torch.optim import Adam
import matplotlib.pyplot as plt
import math

from src.utils.data import load_data
from src.utils.priors import IndependentPrior, CevPrior, NigPrior
from src.utils.distributions import ScaledBeta

from src.models import Bs, Cev, Nig, Sv, Sabr

"""
Compute the MLE every months for several models.
The log-likelihood of each models at each time is plotted for comparison.

Usage:
    ./run.sh experiments/mle_sp500.py
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()

def mle(model, params_init, data, lr, start, stop,
        optimization_times, n_grad_steps, window):
    """
    Compute the MLE of a model using historical data at several point in time.
    """
    params = model.transform.inv(params_init).requires_grad_(True)
    optimizer = Adam([params], lr=lr)

    log_lik = torch.zeros(size=(stop-start,))
    for t in range(start, stop):
        print(f'Day {t} / {stop-1}')
        current_data = data[t-window:t]

        if t in optimization_times:
            for _ in range(n_grad_steps):
                optimizer.zero_grad()
                loss = -model.ll(params, current_data)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            log_lik[t-start] = model.ll(params, current_data)
    return log_lik

def main(args):
    torch.manual_seed(args.seed)

    # === Load market data ===
    path = 'data/spx_spot.csv'
    dates, S = load_data(path)

    # === Hyper parameters ===
    dt = 1 / 252
    window = 252                # Use last year of data to compute the MLE
    optimization_freq = 20      # Compute the MLE every 20 days
    start = window
    stop = len(S) + 1
    optimization_times = torch.arange(start, stop, step=optimization_freq)
    n_grad_steps = 50           # Do 50 gradient steps to compute the MLE

    # === Hyper parameters ===
    # Need to define a prior to map the parameters to an unconstrained space.
    prior = IndependentPrior([
        D.Normal(0., 1.),
        D.LogNormal(0., 1.)
    ])
    bs_model = Bs(dt, prior)
    bs_init = torch.tensor([[0.01, 0.2]])   # Initial guess: mu, sigma

    prior = CevPrior(
        mu_dist = D.Uniform(-0.5, 0.5),
        beta_dist = ScaledBeta(5., 5., low=torch.tensor(0.5), high=torch.tensor(2.0)),
        v = 0.2, S=1000
    )
    cev_model = Cev(dt, prior)
    cev_init = torch.tensor([[0.01, 10.0, 1.0]])

    prior = NigPrior(
        mu_dist=D.Normal(0., 1.),
        sigma_dist=D.LogNormal(0., 1.),
        theta_eta=-math.log(0.01) / 0.1,
        theta_xi=-math.log(0.001) / 5.
    )
    nig_model = Nig(dt, prior)
    nig_init = torch.tensor([[0.0, 0.2, -0.05, 0.05]])

    prior = IndependentPrior([
        D.Normal(0., 1.),
        D.LogNormal(0., 1.),
        D.LogNormal(0., 1.),
        D.Uniform(-1., 1.),
        D.Uniform(-1., 1.)
    ])
    sv_model = Sv(dt, prior)
    sv_init = torch.tensor([[0.0, 0.15, 4.0, 0.95, -0.7]])

    prior = IndependentPrior([
        D.Normal(0., 1.),       # mu, beta, sigma, rho
        D.LogNormal(0., 1.),
        D.LogNormal(0., 1.),
        D.Uniform(-1., 1.)
    ])
    sabr_model = Sabr(dt, prior)
    sabr_init = torch.tensor([[0.0, 1.0, 0.2, -0.5]])

    # === Compute the MLE for each model at each time ===
    bs_ll = mle(bs_model, bs_init, S, 0.1, start, stop, optimization_times, n_grad_steps, window)
    cev_ll = mle(cev_model, cev_init, S, 0.1, start, stop, optimization_times, n_grad_steps, window)
    nig_ll = mle(nig_model, nig_init, S, 0.1, start, stop, optimization_times, n_grad_steps, window)
    sv_ll = mle(sv_model, sv_init, S, 0.01, start, stop, optimization_times, n_grad_steps, window)
    sabr_ll = mle(sabr_model, sabr_init, S, 0.01, start, stop, optimization_times, n_grad_steps, window)

    # === Plot the log-likelihood value at the MLE for each model at each time ===
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates[start-1:stop], bs_ll, label='bs', linewidth=0.8)
    ax.plot(dates[start-1:stop], cev_ll, label='cev', linewidth=0.8)
    ax.plot(dates[start-1:stop], nig_ll, label='nig', linewidth=0.8)
    ax.plot(dates[start-1:stop], sv_ll, label='sv', linewidth=0.8)
    ax.plot(dates[start-1:stop], sabr_ll, label='sabr', linewidth=0.8)
    ax.legend()

    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
