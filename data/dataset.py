#%%
import sys
sys.path.append('../')

from torch.utils.data import Dataset
import h5py
from omegaconf import OmegaConf
import torch
import matplotlib.pyplot as plt

# Load configuration from YAML
config_path = "/Users/timnaher/Documents/PhD/Projects/SODeep/configs/data/data_config.yaml"
cfg = OmegaConf.load(config_path)


import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import savgol_filter

class SOTimeSeriesDataset(Dataset):
    def __init__(self, hdf5_file, label_time, transform=None, derivative=False, window_length=10, polyorder=2):
        """
        Custom dataset for SO time series data with Savitzky-Golay filtering.

        Args:
            hdf5_file (str): Path to the HDF5 file containing data.
            label_time (str): Name of the label dataset in the HDF5 file.
            transform (callable, optional): Transformation to apply to the data.
            derivative (bool, optional): Whether to calculate the derivative of the data.
            window_length (int, optional): The length of the window for the Savitzky-Golay filter.
            polyorder (int, optional): The polynomial order for the Savitzky-Golay filter.
        """
        self.hdf5_file = hdf5_file
        self.label_time = label_time
        self.transform = transform
        self.derivative = derivative
        self.window_length = window_length
        self.polyorder = polyorder
        
        with h5py.File(self.hdf5_file, 'r') as hdf:
            self.length = hdf['data'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as hdf:
            data = hdf['data'][idx]
            label = hdf[f'labels{self.label_time}'][idx]

        # Apply Savitzky-Golay filter to smooth the data
        data = savgol_filter(data, window_length=self.window_length, polyorder=self.polyorder,deriv=1, axis=-1)

        # Convert to PyTorch tensors
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # Compute the derivative and append it as a new feature if requested
        if self.derivative:
            derivative = torch.diff(data, prepend=data[:1])  # Compute the derivative
            data_with_derivative = torch.stack([data, derivative], dim=0)
        else:
            data_with_derivative = data

        # Apply the transformation if provided
        if self.transform:
            data_with_derivative = self.transform(data_with_derivative) if self.derivative else self.transform(data)

        # Return the data and label
        if self.derivative:
            return data_with_derivative, label
        else:
            # Add a dimension for the channel
            data = data.unsqueeze(0)
            return data, label

#%% Test the dataset
if __name__ == '__main__':
    
    from utils.transforms import population_zscore_transform


    dataset = SOTimeSeriesDataset(
        hdf5_file = cfg.data.test_hdf5_file,
        label_time = -5,
        transform = population_zscore_transform(
            cfg.transforms.z_score_mu,
            cfg.transforms.z_score_sigma
        ))


    # get an element from the dataset

    plt.plot(dataset[23][0].numpy().T)

    # get the mean and std of the data in the dataset
    stds, means = [], []
    for i in range(len(dataset)):
        stds.append(dataset[i][0].std().item())
        means.append(dataset[i][0].mean().item())
    print(f"Mean: {np.mean(means):.2f}, Std: {np.mean(stds):.2f}")
# %%
