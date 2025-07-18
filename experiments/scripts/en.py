import argparse

import torch
from torch import distributions as D
import matplotlib.pyplot as plt
import math

from utils.data import load_data
from utils.priors import IndependentPrior, CevPrior
from utils.distributions import ScaledBeta

from models import Bs, Cev, Nig, Sv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()

def backtest(model, data, u_init, start, stop,
             window, optimization_times, n_particles,
             lr, n_steps, verbose):
    if not start in optimization_times:
        return "Start must be in optimization times."

    n_params = len(u_init[0, :])

    log_lik = torch.zeros(size=(stop-start, n_particles))
    mle_hist = torch.zeros(size=(stop-start, n_params))
    particle_hist = torch.zeros(size=(stop-start, n_particles, n_params))
    for t in range(start, stop):
        if verbose: print(f'Day {t} / {stop-1}')

        current_data = data[t-window:t]
        prior_data = data[t - 2*window:t-window]

        if t in optimization_times:             # Update the model pool
            u_mle, u_particles = model.params_uncertainty(
                u_init, prior_data, n_particles, lr, n_steps, verbose=False
            )
            u_init = u_mle # Start next iteration at previous MLE

        c_mle = model.transform.to(u_mle)
        mle_hist[t-start, :] = c_mle
        c_particles = model.transform.to(u_particles)
        particle_hist[t-start, :] = c_particles

        with torch.no_grad():
            log_lik[t-start, :] = model.ll(u_particles, current_data)
    return log_lik, mle_hist, particle_hist

def main(args):
    torch.manual_seed(args.seed)

    ######## Load data ########
    path = 'data/spx_spot.csv'
    dates, S = load_data(path)
    #dates, S = load_data(path, start='2006-01-01', end='2012-12-31')

    ######## Hyper parameters ########
    dt = 1 / 252
    window = 252
    optimization_freq = 20 # days
    start = 2 * window
    stop = len(S) + 1
    optimization_times = torch.arange(start, stop, step=optimization_freq)
    n_particles = [10, 10, 10, 10]
    n_models = 4

    ######## Instantiate the model ########
    prior = IndependentPrior([  # Only used to constain the params
        D.Normal(0., 1.0),
        ScaledBeta(2.0, 2.0, low=0.01, high=1.0)
    ])
    bs_model = Bs(dt, prior)
    bs_init = torch.tensor([[0.01, 0.2]])   # mu, sigma
    bs_u_init = bs_model.transform.inv(bs_init)

    bs_ll, bs_mles, bs_particles = backtest(bs_model, S, bs_u_init,
                                                start, stop, window, optimization_times,
                                                n_particles[0], lr=0.1, n_steps=100, verbose=args.verbose)

    prior = CevPrior(
        mu_dist = D.Normal(0., 1.),
        beta_dist = ScaledBeta(5., 5., low=torch.tensor(0.5), high=torch.tensor(2.0)),
        v = 0.2, S=1000
    )
    cev_model = Cev(dt, prior)
    cev_init = torch.tensor([[0.01, 10.0, 1.0]])
    cev_u_init = cev_model.transform.inv(cev_init)

    cev_ll, cev_mles, cev_particles = backtest(cev_model, S, cev_u_init,
                                                start, stop, window, optimization_times,
                                                n_particles[1], lr=0.5, n_steps=100, verbose=args.verbose)

    prior = IndependentPrior([
        D.Normal(0., 1.0),
        ScaledBeta(2.0, 2.0, low=0.01, high=1.0),
        D.Normal(-2.0, 2.0),
        ScaledBeta(1.5, 5.0, low=1e-6, high=0.1)
    ])
    nig_model = Nig(dt, prior)
    nig_init = torch.tensor([[0.0, 0.2, -1.0, 0.05]])
    nig_u_init = nig_model.transform.inv(nig_init)
    nig_ll, nig_mles, nig_particles = backtest(nig_model, S, nig_u_init,
                                                start, stop, window, optimization_times,
                                                n_particles[2], lr=0.5, n_steps=100, verbose=args.verbose)

    prior = IndependentPrior([
        D.Normal(0., 0.1),
        D.LogNormal(math.log(1.0) - 0.5 * 1.0**2, 1.0),
        D.LogNormal(math.log(5.0) - 0.5 * 0.8**2, 0.8),
        ScaledBeta(3.0, 1.5, -1.0, 1.0),
        ScaledBeta(1.5, 3.0, -1.0, 1.0)
    ])
    sv_model = Sv(dt, prior)
    sv_init = torch.tensor([[0.0, 0.15, 4.0, 0.95, -0.7]])
    sv_u_init = sv_model.transform.inv(sv_init)
    sv_ll, sv_mles, sv_particles = backtest(sv_model, S, sv_u_init,
                                                start, stop, window, optimization_times,
                                                n_particles[3], lr=0.1, n_steps=200, verbose=args.verbose)

    fig1, ax = plt.subplots(figsize=(10, 4))
    #ax.plot(dates[start-1:stop], 10 * S[start-1:stop] / S[0], linewidth=0.5, c='grey')
    ax.plot(dates[start-1:stop], bs_ll, linewidth=0.5, c='red')
    ax.plot(dates[start-1:stop], cev_ll, linewidth=0.5, c='blue')
    ax.plot(dates[start-1:stop], nig_ll, linewidth=0.5, c='green')
    ax.plot(dates[start-1:stop], sv_ll, linewidth=0.5, c='pink')
    ax.set_title(label='Particle log-likelihoods')

    n_params_model = [2, 3, 4, 5]
    particles = [bs_particles, cev_particles, nig_particles, sv_particles]
    for m in range(n_models):
        fig = plt.figure(figsize=(10, 4))
        n_params = n_params_model[m]
        part = particles[m]
        for i in range(n_params):
            plt.subplot(n_params + 1, 1, i+1)
            for n in range(n_particles[m]):
                plt.scatter(dates[start-1:stop], part[:, n, i], marker='+', s=1.0)
            plt.grid()
        fig.tight_layout()

    all_ll = torch.cat([bs_ll, cev_ll, nig_ll, sv_ll], dim=1)       # (t, all particles from all models at time t)
    log_norm_constant = torch.logsumexp(all_ll, dim=1, keepdim=True)
    log_bs_pw = bs_ll - log_norm_constant       # pw = particle weight
    log_cev_pw = cev_ll - log_norm_constant
    log_nig_pw = nig_ll - log_norm_constant
    log_sv_pw = sv_ll - log_norm_constant

    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates[start-1:stop], log_bs_pw, linewidth=0.5, c='red')
    ax.plot(dates[start-1:stop], log_cev_pw, linewidth=0.5, c='blue')
    ax.plot(dates[start-1:stop], log_nig_pw, linewidth=0.5, c='green')
    ax.plot(dates[start-1:stop], log_sv_pw, linewidth=0.5, c='pink')
    ax.set_title(label='Particle log weights')

    log_bs_mw = torch.logsumexp(log_bs_pw, dim=1)       # mw = model weight
    log_cev_mw = torch.logsumexp(log_cev_pw, dim=1)
    log_nig_mw = torch.logsumexp(log_nig_pw, dim=1)
    log_sv_mw = torch.logsumexp(log_sv_pw, dim=1)
    #all_log_mw = torch.cat([log_bs_mw.unsqueeze(1), log_cev_mw.unsqueeze(1), log_nig_mw.unsqueeze(1), log_sv_mw.unsqueeze(1)], dim=1)
    #print(torch.logsumexp(all_log_mw, dim=1))       # Sanity check: All ap zeros = total weight is one

    fig3, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates[start-1:stop], log_bs_mw, linewidth=0.5, label='bs', c='red')
    ax.plot(dates[start-1:stop], log_cev_mw, linewidth=0.5, label='cev', c='blue')
    ax.plot(dates[start-1:stop], log_nig_mw, linewidth=0.5, label='nig', c='green')
    ax.plot(dates[start-1:stop], log_sv_mw, linewidth=0.5, label='sv', c='pink')
    ax.set_title(label='Model log weights')
    fig3.legend()

    fig4, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates[start-1:stop], log_bs_mw.exp(), linewidth=0.5, label='bs', c='red')
    ax.plot(dates[start-1:stop], log_cev_mw.exp(), linewidth=0.5, label='cev', c='blue')
    ax.plot(dates[start-1:stop], log_nig_mw.exp(), linewidth=0.5, label='nig', c='green')
    ax.plot(dates[start-1:stop], log_sv_mw.exp(), linewidth=0.5, label='sv',  c='pink')
    ax.set_title(label='Model weights')
    fig4.legend()

    plt.show()
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)