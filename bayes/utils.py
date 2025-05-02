import torch

def likelihood_with_updates(model, optimizer, optimization_times, n_grad_steps, spot_prices, window, logging=None):
    T = len(spot_prices)
    log_l = torch.zeros(size=(T,))
    for t in range(1, T):
        if t % 100 == 0: print(t)
        if t in optimization_times:
            for j in range(n_grad_steps):
                optimizer.zero_grad()
                loss = - model(spot_prices, t=t, window=window)
                loss.backward()
                optimizer.step()
                if logging is not None:
                    params = model.inv_reparam()
                    params_names = model.params_names
                    string = [f"{k}: {v.item():.3f}" for k, v in zip(params_names, params)]
                    logging.info(f"t: {t}, " + ", ".join(string))

        with torch.no_grad():
            if t >= 100 and model.model_type == 'SV':
                log_l[t] = model(spot_prices, t=t, window=window, update_v0=True)
            else:
                log_l[t] = model(spot_prices, t=t, window=window)
    return log_l
