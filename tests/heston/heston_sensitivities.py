import torch
import numpy as np
import matplotlib.pyplot as plt

from models.heston_class import Heston

torch.manual_seed(42)

dts = torch.tensor([1 / 252, 10 / 252])
n_params = 5

s0 = torch.tensor(100.0)
v0 = torch.tensor(0.3**2)
s_next = torch.linspace(s0 - 20, s0 + 20, 100)

mu_true = torch.tensor(0.01)
k_true = torch.tensor(4.0)
theta_true = torch.tensor(0.04)
sigma_true = torch.tensor(0.5)
rho_true = torch.tensor(-0.8)

fig, axs = plt.subplots(nrows=len(dts), ncols=n_params, figsize=(10, 6))
cs = ['blue', 'green']
for i, dt in enumerate(dts):
    mus = torch.tensor([0.1, 1.0])
    for j, mu in enumerate(mus):
        model = Heston(mu, k_true, theta_true, sigma_true, rho_true, v0)
        transition = model.mc_transition(s0, v0, s_next, dt)
        axs[i, 0].plot(torch.log(s_next), transition.detach(), label=f'mu={mu.item():.2f}', linewidth=0.8, c=cs[j])

        exact_transition = model.ft_transition(s0, v0, s_next, dt)
        label = 'exact' if j==0 else ''
        axs[i, 0].plot(torch.log(s_next), exact_transition.detach(), label=label, c='orange', linestyle='--', linewidth=0.8)

    ks = torch.tensor([1.0, 10.0])
    for j, k in enumerate(ks):
        model = Heston(mu_true, k, theta_true, sigma_true, rho_true, v0)
        transition = model.mc_transition(s0, v0, s_next, dt)
        axs[i, 1].plot(torch.log(s_next), transition.detach(), label=f'k={k.item():.2f}', linewidth=0.8, c=cs[j])

        exact_transition = model.ft_transition(s0, v0, s_next, dt)
        label = 'exact' if j==0 else ''
        axs[i, 1].plot(torch.log(s_next), exact_transition.detach(), label=label, c='orange', linestyle='--', linewidth=0.8)

    thetas = torch.tensor([0.1**2, 1.0**2])
    for j, theta in enumerate(thetas):
        model = Heston(mu_true, k_true, theta, sigma_true, rho_true, v0)
        transition = model.mc_transition(s0, v0, s_next, dt)
        axs[i, 2].plot(torch.log(s_next), transition.detach(), label=f'theta={theta.item():.2f}', linewidth=0.8, c=cs[j])

        exact_transition = model.ft_transition(s0, v0, s_next, dt)
        label = 'exact' if j==0 else ''
        axs[i, 2].plot(torch.log(s_next), exact_transition.detach(), label=label, c='orange', linestyle='--', linewidth=0.8)

    sigmas = torch.tensor([0.1, 1.5])
    for j, sigma in enumerate(sigmas):
        model = Heston(mu_true, k_true, theta_true, sigma, rho_true, v0)
        transition = model.mc_transition(s0, v0, s_next, dt)
        axs[i, 3].plot(torch.log(s_next), transition.detach(), label=f'sigma={sigma.item():.2f}', linewidth=0.8, c=cs[j])

        exact_transition = model.ft_transition(s0, v0, s_next, dt)
        label = 'exact' if j==0 else ''
        axs[i, 3].plot(torch.log(s_next), exact_transition.detach(), label=label, c='orange', linestyle='--', linewidth=0.8)

    rhos = torch.tensor([-0.8, 0.8])
    for j, rho in enumerate(rhos):
        model = Heston(mu_true, k_true, theta_true, sigma_true, rho, v0)
        transition = model.mc_transition(s0, v0, s_next, dt)
        axs[i,4].plot(torch.log(s_next), transition.detach(), label=f'rho={rho.item():.2f}', linewidth=0.8, c=cs[j])
    
        exact_transition = model.ft_transition(s0, v0, s_next, dt)
        label = 'exact' if j==0 else ''
        axs[i, 4].plot(torch.log(s_next), exact_transition.detach(), label=label, c='orange', linestyle='--', linewidth=0.8)

for j in range(n_params):
    axs[0, j].legend()
    
fig.suptitle(f'Heston sensitivities over {int(252 * dts[0])} and {int(252 * dts[1])} days. Exact and approximated densities.', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
