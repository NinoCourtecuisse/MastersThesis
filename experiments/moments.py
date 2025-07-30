import torch
import matplotlib.pyplot as plt
from src.utils.data import load_data

def main():
    ######## Load data ########
    path = 'data/spx_spot.csv'
    dates, s = load_data(path)

    log_returns = torch.log(s[1:] / s[:-1])
    ######## Compute moments ########
    mean = log_returns.mean()
    std = log_returns.std()
    min = log_returns.min()
    max = log_returns.max()

    centered = log_returns - mean
    skewness = (centered**3).mean() / (std**3)
    kurtosis = (centered**4).mean() / (std**4)
    excess_kurtosis = kurtosis - 3

    print(f"Mean: {mean.item():.4f}")
    print(f"Std: {std.item():.4f}")
    print(f"Skewness: {skewness.item():.4f}")
    print(f"Excess Kurtosis: {excess_kurtosis.item():.4f}")
    print(f"Min: {min.item():.4f}")
    print(f"Max: {max.item():.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(dates[1:], log_returns, linewidth=0.8)
    plt.show()

if __name__ == '__main__':
    main()
