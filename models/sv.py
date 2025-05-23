from abc import abstractmethod
from models.model_class import Model
import torch

class Sv(Model):
    def __init__(self, params, params_names):
        super().__init__(params, params_names)
        self.model_type = 'SV'

    @abstractmethod
    def variance_path(self, spot_prices, v0, delta_t):
        pass

    @abstractmethod
    def forward(self, spot_prices, variance_path, delta_t, window=100):
        pass
    
    def likelihood_with_updates(self, v0, optimizer, optimization_times, n_grad_steps, \
                                spot_prices, dt, window, logging=None, verbose=False):
        T = len(spot_prices)
        log_l = torch.zeros(size=(T,))

        # Generate a variance path with the init params
        var_path = self.variance_path(spot_prices[:optimization_times[0]], v0, dt)
        prev_optimization_time = window
        for t in range(window, T):
            if verbose and t % 100 == 0: print(t)
            
            if t in optimization_times:
                for i in range(n_grad_steps):
                    var_path = self.variance_path(spot_prices[t-window:t+1],
                                                  v0=var_path[0].detach(), delta_t=dt)

                    optimizer.zero_grad()
                    loss = - self(spot_prices[t-window:t+1], var_path, dt, window)
                    loss.backward()
                    optimizer.step()    # Update params
                
                idx = torch.where(optimization_times == t)[0]
                start = t - window
                if idx != len(optimization_times) - 1:
                    stop = optimization_times[idx + 1]
                else:
                    stop = T - 1
                prev_optimization_time = optimization_times[idx]
                var_path = self.variance_path(spot_prices[start:stop], var_path[start], dt) # Generate new variance path

            with torch.no_grad():
                log_l[t] = self(spot_prices[t-window:t+1], var_path[t-prev_optimization_time:t-prev_optimization_time+window+1], dt, window)
        return log_l

