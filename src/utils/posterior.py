import torch
import numpy as np

def posterior_mean(particles, weights):
    mean = (weights.unsqueeze(1) * particles).sum(dim=0)
    return mean

def posterior_quantile(particles, normalized_weights, alpha):
    n_params = particles.shape[1]
    quantiles = np.zeros((n_params,))
    for i in range(n_params):
        quantiles[i] = np.quantile(particles[:, i], q=alpha, weights=normalized_weights, method='inverted_cdf')
    return torch.tensor(quantiles)

def summary(particles, weights):
    # Posterior mean
    mean = posterior_mean(particles, weights)

    # 95% credible interval (chosen to be the equal-tailed interval)
    tau = 0.95
    q1 = posterior_quantile(particles, weights, alpha=(1-tau)/2)
    q2 = posterior_quantile(particles, weights, alpha=(1+tau)/2)
    
    return mean, q1, q2

"""
def posterior_quantile(particles, normalized_weights, alpha):
    n_params = particles.shape[1]
    quantiles = torch.zeros(size=(n_params,))
    for i in range(n_params):
        sorter = torch.argsort(particles[:, i])
        values, w = particles[sorter, i], normalized_weights[sorter]
        cum_w = torch.cumsum(w)

        idx = torch.searchsorted(cum_w, quantiles, right=True).clamp(max=n_params - 1)
        prev_idx = (idx - 1).clamp(min=0)

        x0 = cum_w[prev_idx]
        x1 = cum_w[idx]
        y0 = values[prev_idx]
        y1 = values[idx]

        denom = (x1 - x0).clamp(min=1e-12)
        quantiles[i] = y0 + (quantiles - x0) * (y1 - y0) / denom
    return quantiles
"""
