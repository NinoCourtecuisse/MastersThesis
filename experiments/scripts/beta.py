import torch
from utils.distributions import ScaledBeta
import matplotlib.pyplot as plt

def main():
    eval = torch.linspace(-10, 10, 100)

    alpha = [1.2, 2.0]
    beta = [1.2, 2.0]
    for i in range(len(alpha)):
        d = ScaledBeta(alpha[i], beta[i], low=torch.tensor(-10), high=torch.tensor(10))
        pdf = d.log_prob(eval).exp()
        plt.plot(eval, pdf, label=rf'$\alpha$={alpha[i]}, $\beta$={beta[i]}')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
