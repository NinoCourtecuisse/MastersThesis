import torch
from src.utils.distributions import ScaledBeta
import matplotlib.pyplot as plt

def main():
    eval = torch.linspace(-1, 1, 100)

    alpha = [1.2, 2.0, 5.0]
    beta = [1.2, 2.0, 5.0]
    plt.figure()
    for i in range(len(alpha)):
        d = ScaledBeta(alpha[i], beta[i], low=torch.tensor(-1), high=torch.tensor(1))
        pdf = d.log_prob(eval).exp()
        plt.plot(eval, pdf, label=rf'$\alpha$={alpha[i]}, $\beta$={beta[i]}')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
