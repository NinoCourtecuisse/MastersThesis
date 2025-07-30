import torch
import itertools
from inference.sgld import SGLD, PowerLRScheduler

def get_batch_at_index(dataloader, i):
    if i < 0 or i >= len(dataloader):
        raise IndexError(f"Batch index {i} out of range (0 ≤ i < {len(dataloader)})")
    return next(itertools.islice(dataloader, i, i+1))

def compute_log_weights(model_classes, particles, model_classes_log_prior, price_data,
                        temperature=1.0):
    """
    Compute the weights given by "Estimate Nothing", i.e. pi_t for t=t_{k-1}+1, ..., t_k.

    particles = list of tensors for each model class, in the unconstrained parametrization
    model_classes_prior = Tensor of shape (M,)
    price_data = {s_j} for j=t_{k-1}, t_{k-1}+1, ..., t_k
    """
    M = len(model_classes)      # Number of model classes.
    N = len(particles[0])       # Number of particles for model class M_m. Assumed to be constant for all model classes.
    T = len(price_data) - 1

    # Compute the log-likelihoods
    ll = torch.zeros(size=(M, T, N))
    for m in range(M):
        model_class = model_classes[m]
        theta = particles[m]
        log_transitions = model_class.log_transition(theta, price_data[:-1], price_data[1:]).T    # (T, N)
        ll[m, :, :] = torch.cumsum(log_transitions, dim=0)
    ll = ll * temperature

    # Compute the weights
    log_numerator = ll + model_classes_log_prior.view(M, 1, 1)
    log_denomenator = torch.logsumexp(log_numerator, dim=[0, 2], keepdim=True)    # (1, T, 1)
    log_pi = log_numerator - log_denomenator

    # Compute importance weights at t=t_k (only useful for the resampling step)
    log_iw = torch.zeros(size=(M, N))
    log_numerator = ll[:, -1, :]    # (M, N)
    log_denomenator = torch.logsumexp(log_numerator, dim=1, keepdim=True)   # (M, 1)
    log_iw = log_numerator - log_denomenator

    return log_pi, log_iw, ll + model_classes_log_prior.view(M, 1, 1)

def update_particles(model_classes, particles:list[torch.Tensor], log_iw:torch.Tensor,
                   dataloader:torch.utils.data.DataLoader, batch_weights: torch.Tensor,
                   n_grad_steps:int, lrs=list[float], resample:bool=True, 
                   move:bool=True, verbose:bool=False):
    """
    Two step procedure to move the particles:
    1. Resample according the importance weights
    2. Stochastic optimization via SGLD

    batched_price_data: (n_batch, batch_size)
    batch_weights: probability weight assigned to each batch
    """
    M = len(model_classes)
    N = len(particles[0])
    n_batch = len(dataloader)
    #(n_batch, batch_size) = batched_price_data.shape

    new_particles = [None] * M
    for m in range(M):
        model_class = model_classes[m]
        current_particles = particles[m]

        if resample:
            # 1. Resample
            iw_m = log_iw[m, :].exp()
            indices = torch.multinomial(iw_m, num_samples=N, replacement=True)
            current_particles = current_particles[indices, :]

        if move:
            # 2. Move
            params = current_particles.detach().clone().requires_grad_(True)    # Make sure to start from a fresh tree
            optimizer = SGLD(params=[params], lr=lrs[m])
            scheduler = PowerLRScheduler(optimizer, gamma=0.51, lr_min=lrs[m]/10, n_steps=10**4)
            for s in range(n_grad_steps):
                optimizer.zero_grad()

                # Choose batch of data
                batch_idx = torch.multinomial(batch_weights, num_samples=1)
                batch_prices = get_batch_at_index(dataloader, batch_idx)[0]
                #batch_prices = next(itertools.islice(dataloader, batch_idx, batch_idx+1))
                #prices = batched_price_data[batch_idx, :].squeeze()

                # SGLD
                batch_ll = torch.sum(model_class.log_transition(params, batch_prices[:-1], batch_prices[1:]), dim=1)
                batch_lprior = model_class.prior.log_prob(model_class.transform.to(params))
                loss = -(batch_lprior + n_batch * batch_ll).sum()
                loss.backward()
                if verbose and s%10==0: print(f"Step {s}: Loss={loss.item():.3f}, Grad norm={torch.norm(params.grad).item():.3f}")

                optimizer.step()
                scheduler.step()
            #for ep in range(n_epochs):
            #    for batch_prices in dataloader:
            #        optimizer.zero_grad()
#
            #        # SGLD
            #        batch_prices = batch_prices[0]
            #        batch_ll = torch.sum(model_class.log_transition(params, batch_prices[:-1], batch_prices[1:]), dim=1)
            #        batch_lprior = model_class.prior.log_prob(model_class.transform.to(params))
            #        loss = -(batch_lprior + n_batch * batch_ll).sum()
            #        loss.backward()
            #        if verbose: print(f"Epoch {ep+1}: Loss={loss.item():.3f}, Grad norm={torch.norm(params.grad).item():.3f}")
#
            #        optimizer.step()
            #        scheduler.step()
            current_particles = params.detach().clone()

        new_particles[m] = current_particles
    return new_particles
