import matplotlib.pyplot as plt
import torch

from models.heston_class import Heston
torch.manual_seed(42)

dt = torch.tensor(1/252, requires_grad=False)
mu = torch.tensor(0.01)
k = torch.tensor(1.0)
theta = torch.tensor(0.04)
rho = torch.tensor(-0.8)
v0 = torch.tensor(0.04)

s0 = torch.tensor(100.0)
s_next = torch.linspace(95.0, 105.0, 100)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,4))

sigma = torch.tensor(0.4)
heston_model = Heston(mu, k, theta, sigma, rho, v0)
ft_transition = heston_model.ft_transition(s0, v0, s_next, dt)
mc_transition = heston_model.mc_transition(s0, v0, s_next, dt)
euler_transition = heston_model.euler_transition(s0, v0, s_next, dt)

axs[0].plot(torch.log(s_next), ft_transition.detach(), label='cf', linewidth=0.8)
axs[0].plot(torch.log(s_next), mc_transition.detach(), label='mc', linewidth=0.8)
axs[0].plot(torch.log(s_next), euler_transition.detach(), label='euler', linewidth=0.8)

sigma = torch.tensor(1.0)
heston_model = Heston(mu, k, theta, sigma, rho, v0)
ft_transition = heston_model.ft_transition(s0, v0, s_next, dt)
mc_transition = heston_model.mc_transition(s0, v0, s_next, dt)
euler_transition = heston_model.euler_transition(s0, v0, s_next, dt)

axs[1].plot(torch.log(s_next), ft_transition.detach(), label='cf', linewidth=0.8)
axs[1].plot(torch.log(s_next), mc_transition.detach(), label='mc', linewidth=0.8)
axs[1].plot(torch.log(s_next), euler_transition.detach(), label='euler', linewidth=0.8)

for i in range(2):
    axs[i].grid()
    axs[i].legend()
    axs[i].set_xlabel(r'$X_{t+\Delta}$')
fig.tight_layout()
plt.savefig('figures/heston_transitions.png')
