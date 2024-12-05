#%%
from torchvision import transforms
import torch
import pandas as pd
import numpy as np
import os


#data_path = "/Volumes/T7/SchemAcS/labeled_windows/schemacs_sleep_params.csv"
#data = pd.read_csv(data_path)
#mu   = np.mean(data['mean_amplitude_N2N3'])
#sig  = np.mean(data['sd_amplitude_N2N3'])


def population_zscore_transform(mu, sig):
    """
    Creates a normalization transform for 1D time series data.

    Args:
        mu (float): Mean for normalization.
        sig (float): Standard deviation for normalization.

    Returns:
        Callable: A transformation pipeline.
    """
    return transforms.Compose([
    transforms.Lambda(lambda x: (x - mu) / sig),  # Normalize using mean and std
])




# %%
