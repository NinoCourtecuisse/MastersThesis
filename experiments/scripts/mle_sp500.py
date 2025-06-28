import argparse

import torch
from torch import distributions as D
from torch.optim import Adam
import matplotlib.pyplot as plt

from utils.data import load_data
from utils.priors import IndependentPrior

from models import Bs, Cev, Nig, Sv, Sabr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()

def mle(model, params_init, data, lr, start, stop,
        optimization_times, n_grad_steps, window, verbose=False):
    params = model.transform.inv(params_init).requires_grad_(True)
    optimizer = Adam([params], lr=lr)

    log_lik = torch.zeros(size=(stop-start,))
    for t in range(start, stop):
        if verbose: print(f'Day {t} / {stop-1}')
        current_data = data[t-window:t]
        if isinstance(model, Sv) or isinstance(model, Sabr):
            model.build_objective(current_data)

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

    ######## Load data ########
    path = 'data/spx_spot.csv'
    dates, S = load_data(path)

    ######## Hyper parameters ########
    dt = 1 / 252
    window = 200
    optimization_freq = 20 # days
    start = window
    stop = len(S) + 1
    optimization_times = torch.arange(start, stop, step=optimization_freq)
    n_grad_steps = 50

    ######## Instantiate the model ########
    prior = IndependentPrior([  # Only used to constain the params
        D.Normal(0., 1.),
        D.LogNormal(0., 1.)
    ])
    bs_model = Bs(dt, prior)
    bs_init = torch.tensor([[0.01, 0.2]])   # mu, sigma

    prior = IndependentPrior([
        D.Normal(0., 1.),
        D.LogNormal(0., 1.),
        D.Uniform(0., 2.5)
    ])
    cev_model = Cev(dt, prior)
    cev_init = torch.tensor([[0.01, 10.0, 1.0]])

    prior = IndependentPrior([
        D.Normal(0., 1.),
        D.LogNormal(0., 1.),
        D.Normal(0., 1.),
        D.LogNormal(0., 1.)
    ])
    nig_model = Nig(dt, prior)
    nig_init = torch.tensor([[0.01, 0.2, -1.0, 0.01]])

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

    ######## MLE ########

    bs_ll = mle(bs_model, bs_init, S, 0.1, start, stop, optimization_times, n_grad_steps, window,
                verbose=args.verbose)
    cev_ll = mle(cev_model, cev_init, S, 0.1, start, stop, optimization_times, n_grad_steps, window,
                verbose=args.verbose)
    nig_ll = mle(nig_model, nig_init, S, 0.1, start, stop, optimization_times, n_grad_steps, window,
                verbose=args.verbose)
    sv_ll = mle(sv_model, sv_init, S, 0.01, start, stop, optimization_times, n_grad_steps, window,
                verbose=args.verbose)
    sabr_ll = mle(sabr_model, sabr_init, S, 0.01, start, stop, optimization_times, n_grad_steps, window,
                verbose=args.verbose)

    fig, ax = plt.subplots(figsize=(10, 4))
    tau = 0.01
    ax.plot(dates[start-1:stop], 10 * S[start-1:stop] / S[0], linewidth=0.5, c='grey')
    ax.plot(dates[start-1:stop], tau * bs_ll, label='bs', linewidth=0.8)
    ax.plot(dates[start-1:stop], tau * cev_ll, label='cev', linewidth=0.8)
    ax.plot(dates[start-1:stop], tau * nig_ll, label='nig', linewidth=0.8)
    ax.plot(dates[start-1:stop], tau * sv_ll, label='sv', linewidth=0.8)
    ax.plot(dates[start-1:stop], tau * sabr_ll, label='sabr', linewidth=0.8)
    ax.legend()
    plt.show()

    all_ll = torch.stack([bs_ll, cev_ll, nig_ll, sv_ll, sabr_ll], dim=0)
    log_l_normalized = tau * all_ll
    log_normalization = torch.logsumexp(log_l_normalized, dim=0)
    log_posterior = log_l_normalized - log_normalization
    posterior = torch.exp(log_posterior)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates[start-1:stop], 0.5 * S[start-1:stop] / S[0], linewidth=0.5, c='grey')
    ax.plot(dates[start-1:stop], posterior[0, :], label='bs', linewidth=0.8)
    ax.plot(dates[start-1:stop], posterior[1, :], label='cev', linewidth=0.8)
    ax.plot(dates[start-1:stop], posterior[2, :], label='nig', linewidth=0.8)
    ax.plot(dates[start-1:stop], posterior[3, :], label='sv', linewidth=0.8)
    ax.plot(dates[start-1:stop], posterior[4, :], label='sabr', linewidth=0.8)
    ax.grid()
    ax.legend()
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
