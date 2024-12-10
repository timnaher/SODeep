#%%
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
from scipy.signal import savgol_filter
import h5py
from omegaconf import OmegaConf


##########################
##### Custom Dataset #####
##########################

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


##########################
####### Datamodule #######
##########################


class WindowedEEGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        val_test_window_length: Optional[int] = None,
        transforms: dict = None,
    ) -> None:

        super().__init__()
        
        self.window_length = window_length
        self.val_test_window_length = val_test_window_length or window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        # Transforms can be provided as a dictionary for train, val, test
        self.train_transforms = transforms.get("train") if transforms else None
        self.val_transforms = transforms.get("val") if transforms else None
        self.test_transforms = transforms.get("test") if transforms else None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                SOTimeSeriesDataset(
                    hdf5_path,
                    transform=self.train_transforms,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                SOTimeSeriesDataset(
                    hdf5_path,
                    transform=self.val_transforms,
                    window_length=self.val_test_window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                SOTimeSeriesDataset(
                    hdf5_path,
                    transform=self.test_transforms,
                    window_length=self.val_test_window_length,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

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
