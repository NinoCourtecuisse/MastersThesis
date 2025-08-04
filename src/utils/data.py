from typing import Tuple

import torch
import pandas as pd
import datetime
import numpy as np

def load_data(path:str, start:str='', end:str='') -> Tuple[np.ndarray, torch.Tensor]:
    """
    Load time series data from a CSV file with columns ['date', 'close'].
    Optionally filters the data to a date range [start, end].

    Args:
        path (str): Path to the CSV file.
        start (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to '' (no filter).
        end (str, optional): End date in 'YYYY-MM-DD' format. Defaults to '' (no filter).

    Returns:
        Tuple[np.ndarray, torch.Tensor]:
            - `dates`: NumPy array of dates (np.datetime64).
            - `prices`: PyTorch tensor of asset prices (float32).
    """
    spot_data = pd.read_csv(path, sep=',')
    if spot_data.iloc[0, 1] == '...':
        raise ValueError(f"The data was not loaded in {path}. See the Readme for instructions.")
    spot_data['date'] = pd.to_datetime(spot_data['date'])

    if len(start) > 0 and len(end) > 0:
        start = datetime.datetime.strptime(start, '%Y-%m-%d')
        end = datetime.datetime.strptime(end, '%Y-%m-%d')
        spot_data = spot_data[(spot_data['date'] >= start) & (spot_data['date'] <= end)]

    spot_data.set_index('date', inplace=True)
    s = spot_data['close'].to_numpy()

    dates = spot_data.index.to_numpy()
    s = torch.tensor(s, dtype=torch.float32)
    return dates, s
