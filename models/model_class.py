import torch

class Model(torch.nn.Module):
    def __init__(self, params, params_names):
        super().__init__()
        self.params = params
        self.params_names = params_names

    def get_params(self):
        return self.params

    def likelihood_with_updates(self, optimizer, optimization_times, n_grad_steps, spot_prices, window, logging=None, verbose=False):
        T = len(spot_prices)
        log_l = torch.zeros(size=(T,))
        for t in range(1, T):
            if t in optimization_times:
                for _ in range(n_grad_steps):
                    optimizer.zero_grad()
                    loss = - self(spot_prices, t=t, window=window)
                    loss.backward()
                    optimizer.step()
                    if logging is not None:
                        params = self.inv_reparam()
                        params_names = model.params_names
                        string = [f"{k}: {v.item():.3f}" for k, v in zip(params_names, params)]
                        logging.info(f"t: {t}, " + ", ".join(string))