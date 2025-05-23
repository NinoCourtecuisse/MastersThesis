import torch
import matplotlib.pyplot as plt
import numpy as np

from models.nig import Nig

torch.manual_seed(42)
np.random.seed(42)

dt = torch.tensor(1 / 252)
T = 1.0
S0 = torch.tensor(100.0)

mu = torch.tensor(0.0)
alpha = torch.tensor(3.0)
beta = torch.tensor(1.0)
delta = torch.tensor(1.0)

nig_model = Nig(mu, alpha, beta, delta)
mean, std = nig_model.get_moments()
print(f'return mean={mean}, std={std}')
S = nig_model.simulate(S0, dt, T, M=100)

plt.figure(figsize=(7,3))
plt.plot(torch.linspace(0, T, len(S)), S.detach())
plt.xlabel('Time')
plt.ylabel('Spot')
plt.grid()
plt.show()