import torch

def compute_phi(log_prob_grad, inputs, h, h_per_dimension=False):
    n, d = inputs.shape

    if h > 0:
        pairwise_distance = torch.norm(inputs[:, None] - inputs, dim=2).pow(2)
        K = torch.exp( - pairwise_distance / (2 * h))

    else: # median heuristic
        mask = ~torch.eye(n, dtype=torch.bool)

        if not h_per_dimension:
            pairwise_distance = torch.norm(inputs[:, None] - inputs, dim=2).pow(2)  # [n, n]
            median = torch.quantile(pairwise_distance[mask], q=0.5)
            h = median / torch.log(torch.tensor(n + 1.)) + 1e-6
            K = torch.exp( - pairwise_distance / (2 * h))

        else: # Compute different bandwidths h_1, ..., h_d
            sq_diff = (inputs[:, None, :] - inputs[None, :, :]).pow(2)  # [n, n, d]

            mask_flat = mask.view(n * n)
            sq_diff_flat = sq_diff.view(n * n, d)
            sq_diff_off_diag = sq_diff_flat[mask_flat, :]
            h = torch.quantile(sq_diff_off_diag, q=0.5, dim=0) / torch.log(torch.tensor(n + 1.)) + 1e-6

            scaled_sq_diff = sq_diff / (2 * h[None, None, :])
            K = torch.exp(- scaled_sq_diff.sum(dim=2))
            h = h.unsqueeze(0)

    Q = (inputs * K.sum(dim=1, keepdim=True) - K @ inputs) / h
    phi = (K @ log_prob_grad + Q) / n
    return phi
