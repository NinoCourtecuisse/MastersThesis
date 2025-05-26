from abc import ABC, abstractmethod
import torch

class Model(torch.nn.Module, ABC):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model_type = 'General'

    @abstractmethod
    def get_params(self):
        pass

    def print_params(self):
        params = self.get_params()
        string = [f"{k}: {v.item():.3f}" for k, v in params.items()]
        return f", ".join(string)

    def likelihood_with_updates(self, optimizer, optimization_times, n_grad_steps, \
                                spot_prices, dt, decay_coef, window, \
                                logging=None, verbose=False):
        T = len(spot_prices)
        log_l = torch.zeros(size=(T,))
        for t in range(window, T):
            if verbose and t % 100 == 0: print(t)

            if t in optimization_times:
                for _ in range(n_grad_steps):
                    optimizer.zero_grad()
                    loss = - self(spot_prices, t=t, delta_t=dt, decay_coef=decay_coef, window=window)
                    loss.backward()
                    optimizer.step()

                    if logging is not None:
                        logging.info(f"t: {t}, " + self.print_params())

            with torch.no_grad():
                if t >= window and self.model_type == 'SV':
                    log_l[t] = self(spot_prices, t=t, delta_t=dt, decay_coef=decay_coef, window=window, update_v0=True)
                else:
                    log_l[t] = self(spot_prices, t=t, delta_t=dt, decay_coef=decay_coef, window=window)
        return log_l
