import torch
import pandas as pd
import datetime

def load_data(path, start:str='', end:str=''):
    spot_data = pd.read_csv(path, sep=',')
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

def batch_data(data, batch_size):
    T = len(data)
    n_batches = T // batch_size
    trim_len = n_batches * batch_size
    data = data[-trim_len:]  # Trim the head to get full batches
    return data.reshape(n_batches, batch_size)
