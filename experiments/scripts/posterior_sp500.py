import argparse

import torch
from torch import distributions as D
import matplotlib.pyplot as plt
import seaborn as sb
import math

from utils.data import load_data, batch_data
from utils.priors import IndependentPrior, CevPrior, NigPrior
from utils.distributions import ScaledBeta

from inference.ibis import backtest
from inference import KernelSGLD

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['bs', 'cev', 'nig', 'sv', 'sabr'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()

def main(args):
    torch.manual_seed(args.seed)

    ######## Load data ########
    path = 'data/spx_spot.csv'
    dates, s = load_data(path, start='2006-01-03', end='2012-12-31')

    batch_size = 15
    s_batch = batch_data(s, batch_size=batch_size)
    n_batch = s_batch.shape[0]

    ######## Hyper parameters ########
    dt = torch.tensor(1 / 252)
    ESS_rmin = 0.5
    window = 100
    kernel = KernelSGLD(n_steps=100, lr=1e-2, lr_min=1e-4, gamma=0.51)
    n_particles = 500

    ######## Instantiate model ########
    match args.model:
        case 'bs':
            from models import Bs as Model
            prior = IndependentPrior([
                D.Normal(0.0, 0.1),     # mu
                D.Uniform(0.01, 1.5)    # sigma
            ])
        case 'cev':
            from models import Cev as Model
            prior = CevPrior(
                mu_dist=D.Normal(0., 0.1),
                beta_dist=ScaledBeta(5., 5., low=torch.tensor(0.5), high=torch.tensor(2.0))
            )
        case 'nig':
            from models import Nig as Model
            prior = NigPrior(
                mu_dist=D.Normal(0., 0.1),
                sigma_dist=D.LogNormal(math.log(0.2), 1.0),
                theta_eta=-math.log(0.01) / 0.1,
                theta_xi=-math.log(0.001) / 5.
            )
        case _:
            return NotImplementedError
    
    model = Model(dt, prior)

    ######## Run IBIS ########
    hist = backtest(model, kernel, s_batch, s, n_particles, ESS_rmin, window, args.verbose)

    batch_idx = [0, n_batch//2, n_batch-2]
    nrows = len(batch_idx)
    ncols = len(hist['particles'][0][0, :])

    plt.figure(figsize=(10, 4))
    for i in range(nrows):
        particles = hist['particles'][batch_idx[i]+1]
        weights = hist['weights'][batch_idx[i]+1]
        for j in range(ncols):
            index = i * ncols + j + 1
            plt.subplot(nrows, ncols, index)
            sb.histplot(x=particles[:, j].numpy(), weights=weights.numpy(), bins=50, stat="density", kde=True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
