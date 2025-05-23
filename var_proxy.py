import matplotlib.pyplot as plt
import pandas as pd
import datetime
import torch
from models.bs_class import Bs

# Load data
start = datetime.datetime.strptime('2006-01-03', '%Y-%m-%d')
end = datetime.datetime.strptime('2012-12-31', '%Y-%m-%d')

SPOT_PATH = 'data/spx_spot.csv'
spot_data = pd.read_csv(SPOT_PATH, sep=',')
spot_data['date'] = pd.to_datetime(spot_data['date'])
spot_data = spot_data[(spot_data['date'] >= start) & (spot_data['date'] <= end)]
spot_data.set_index('date', inplace=True)

S = spot_data['close'].to_numpy()
dates = spot_data.index.to_numpy()
S = torch.tensor(S, dtype=torch.float32)

T = len(S)
vol = torch.zeros_like(S)
dt = torch.tensor(1/252, requires_grad=False)
window = 100
n_grad_steps = 50

mu = torch.tensor(0.01)
sigma = torch.tensor(0.1)
vol[0] = sigma.item()
bs_model = Bs(mu, sigma)

optimizer = torch.optim.Adam(bs_model.parameters(), lr=0.1)
for t in range(1, T):
    if t % 100 == 0: print(t)

    for _ in range(n_grad_steps):
        optimizer.zero_grad()
        loss = - bs_model.forward(S, t, dt, window)
        loss.backward()
        optimizer.step()

    params = bs_model.inv_reparam()
    vol[t] = params[1].item()

df = pd.DataFrame(vol.numpy(), index=spot_data.index, columns=['volatility'])
df.to_csv('data/spx_vol.csv', index=False)

plt.figure(figsize=(10, 5))
plt.plot(S / S[0])
plt.plot(vol)
plt.show()

#df.to_csv('tensor.csv', index=False)