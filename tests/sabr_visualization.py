import matplotlib.pyplot as plt
import pandas as pd
import datetime
import torch

from models.sabr_class import Sabr

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

dt = torch.tensor(1/252, requires_grad=False)
saving_times = torch.tensor([200, 500, 700, 1000])
#saving_times = torch.tensor([200])
window = 200

mu = torch.tensor(0.0)
beta = torch.tensor(1.0)
delta_0 = torch.tensor(10.0)
sigmas = torch.linspace(0.1, 2.0, 25)
rhos = torch.linspace(-0.99, 0.99, 25)
l_history = torch.zeros(len(sigmas), len(rhos), len(saving_times))
l_history[:, :, 0] = 0.0

T = len(S)

log_l = torch.zeros(size=(T,), requires_grad=False)

for l, sigma in enumerate(sigmas):
    print(l)
    for m, rho in enumerate(rhos):
        model = Sabr(mu, beta, sigma, rho, delta_0)
        for t in saving_times:
            idx = torch.where(saving_times == t)[0][0]

            if t >= window:
                update = True
            else:
                update = False
            l_history[l, m, idx] = model.forward(S, t, window=200, delta_t=dt, update_v0=update)

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=len(saving_times), cols=1,
    vertical_spacing=0.04,
)

for i in range(len(saving_times)):
    date = pd.to_datetime(dates[saving_times[i]])
    if i==1:
        show_scale = True
        show_legend = True
    else:
        show_scale = False
        show_legend = False
    fig.add_trace(
        go.Contour(
            z=l_history[:, :, i].detach(),
            x=rhos,
            y=sigmas,
            contours=dict(
                start=torch.max(l_history[:, :, i]).item() - 100.0,
                end=torch.max(l_history[:, :, i]).item(),
                size=0.1,
                coloring='heatmap',
            ),
            #colorscale='Viridis',
            line=dict(width=0.0),
            showscale=False,
        ), 
        row=i+1, col=1)
    #fig.update_yaxes(title_text="mu", row=i+1, col=1)
    fig.add_annotation(
        text=f"t = {date.month}-{date.year}",
        xref="paper", yref="paper",
        x= - 0.12, y= 1.13 - 1.055 * (i + 0.5)/len(saving_times),  # Adjust to center vertically
        showarrow=False,
        font=dict(size=12),
        align="left"
    )

fig.update_layout(
    width=600,  # Total width of the figure
    height=800,  # Total height of the figure
    #title_text="BS likelihood and particles through time",
    xaxis=dict(
        title='rho',
        title_standoff=0.0,
        range=[-0.95, 0.95],
        title_font=dict(size=12),
    ),
    yaxis=dict(
        title='sigma',
        title_standoff=0.0,
        range=[0.1, 2.0],
        title_font=dict(size=12)
    ),
    legend=dict(
        x=1.0,
        y=1.03,
        xanchor='left',
        yanchor='top',
    ),
    margin=dict(
    t=20,
    b=40,
    l=60,
    r=20,
)
)
fig.show()