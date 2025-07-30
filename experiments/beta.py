import argparse

import torch
from utils.distributions import ScaledBeta
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, help='Path to save the plot.')
    return parser.parse_args()

def main(args):
    eval = torch.linspace(-1, 1, 100)

    alpha = [1.2, 2.0, 5.0]
    beta = [1.2, 2.0, 5.0]
    fig1 = plt.figure()
    for i in range(len(alpha)):
        d = ScaledBeta(alpha[i], beta[i], low=torch.tensor(-1), high=torch.tensor(1))
        pdf = d.log_prob(eval).exp()
        plt.plot(eval, pdf, label=rf'$\alpha$={alpha[i]}, $\beta$={beta[i]}')

    plt.legend()
    if args.save:
        fig1.savefig(fname=f'{args.save}/beta_distrib.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
