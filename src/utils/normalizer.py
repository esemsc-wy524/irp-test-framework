import numpy as np
import torch

class StandardScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

    def transform_tensor(self, x):
        return (x - torch.tensor(self.mean, dtype=x.dtype, device=x.device)) / torch.tensor(self.std, dtype=x.dtype, device=x.device)

    def inverse_transform_tensor(self, x):
        return x * torch.tensor(self.std, dtype=x.dtype, device=x.device) + torch.tensor(self.mean, dtype=x.dtype, device=x.device)
