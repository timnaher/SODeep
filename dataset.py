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
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class WindowedEEGDataset(Dataset):
    def __init__(self, hdf5_file, window_length, transform=None, stride=None, keep_ratio=0.1):
        """
        PyTorch Dataset for windowed access to EEG data stored in HDF5 files.

        Args:
            hdf5_file (str): Path to the HDF5 file.
            window_length (int): Length of each window (number of timepoints).
            transform (callable, optional): Transformation to apply to the data.
            stride (int, optional): Stride between consecutive windows.
                Defaults to `window_length` (non-overlapping windows).
            keep_ratio (float, optional): Fraction of zero-only windows to keep.
        """
        self.hdf5_file = hdf5_file
        self.window_length = window_length
        self.transform = transform
        self.stride = stride or window_length
        self.keep_ratio = keep_ratio

        # Open HDF5 file to determine dataset size
        with h5py.File(hdf5_file, "r") as hdf:
            self.num_timepoints = hdf["eeg"].shape[0]
            self.num_channels = hdf["eeg"].shape[1]
            self.targets = np.array(hdf["targets"][:])  # Use "targets" for labels

        # Precompute windows with balancing
        self.windows = self._compute_windows(keep_ratio=self.keep_ratio)

    def _compute_windows(self, keep_ratio=0.1):
        """
        Compute start and end indices for all windows and balance zero-only windows.
        Remove windows containing breaks in the `time` vector.

        Args:
            keep_ratio (float): Fraction of zero-only windows to keep.

        Returns:
            List[Tuple[int, int]]: A list of tuples (start, end).
        """
        windows = []
        zero_only_windows = []

        with h5py.File(self.hdf5_file, "r") as hdf:
            time = np.array(hdf["time"][:])  # Load the `time` vector

            for start in range(0, self.num_timepoints - self.window_length + 1, self.stride):
                end = start + self.window_length

                # Check if the window contains a break in the `time` vector
                if not np.all(np.diff(time[start:end]) == 1):  # Detect non-consecutive values
                    continue  # Skip this window

                # Add valid window
                windows.append((start, end))

                # Check if the window contains only zeros
                if (self.targets[start:end] == 0).all():
                    zero_only_windows.append((start, end))

        # Randomly sample a subset of zero-only windows
        sampled_zero_windows = random.sample(zero_only_windows, int(len(zero_only_windows) * keep_ratio))

        # Combine sampled zero-only windows with all other windows
        non_zero_windows = [w for w in windows if w not in zero_only_windows]
        balanced_windows = non_zero_windows + sampled_zero_windows

        return balanced_windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Fetch the windowed data and corresponding label for a given index.
        """
        start, end = self.windows[idx]

        with h5py.File(self.hdf5_file, "r") as hdf:
            data = hdf["eeg"][start:end, :]
            label = hdf["targets"][start:end]  # Targets for the window

        # Convert to PyTorch tensors
        data = torch.tensor(data.T, dtype=torch.float32)  # Transpose for [channels, time]
        label = torch.tensor(label, dtype=torch.long)

        # Apply transformation if specified
        if self.transform:
            data = self.transform(data)

        return data, label





##########################
####### Datamodule #######
##########################


import os
from pathlib import Path
from typing import Optional, Sequence, Union
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from utils.transforms import population_zscore_transform

class WindowedEEGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        stride: Optional[int],
        keep_ratio: float,
        batch_size: int,
        num_workers: int,
        train_dir: Union[str, Path],
        val_dir: Union[str, Path],
        test_dir: Union[str, Path],
        val_test_window_length: Optional[int] = None,
        transforms: dict = None,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.stride = stride or window_length
        self.keep_ratio = keep_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.test_dir = Path(test_dir)

        # Transforms can be provided as a dictionary for train, val, test
        self.train_transforms = transforms.get("train") if transforms else None
        self.val_transforms = transforms.get("val") if transforms else None
        self.test_transforms = transforms.get("test") if transforms else None

    def _get_hdf5_files(self, directory: Path) -> Sequence[Path]:
        """Helper function to get all HDF5 files from a directory."""
        return sorted(directory.glob("*.h5"))

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_sessions = self._get_hdf5_files(self.train_dir)
        self.val_sessions = self._get_hdf5_files(self.val_dir)
        self.test_sessions = self._get_hdf5_files(self.test_dir)
        
        self.train_dataset = ConcatDataset(
            [
                WindowedEEGDataset(
                    hdf5_path,
                    transform=self.train_transforms,
                    window_length=self.window_length,
                    stride=self.stride,
                    keep_ratio=self.keep_ratio,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEEGDataset(
                    hdf5_path,
                    transform=self.val_transforms,
                    window_length=self.window_length,
                    stride=self.stride,
                    keep_ratio=self.keep_ratio,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEEGDataset(
                    hdf5_path,
                    transform=self.test_transforms,
                    window_length=self.window_length,
                    stride=self.stride,
                    keep_ratio=self.keep_ratio,
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

if __name__ == '__main__':
    # test
    datamodule = WindowedEEGDataModule(
        window_length=150,
        batch_size=32,
        num_workers=0,
        train_dir="data/processed/train",
        val_dir="data/processed/val",
        test_dir="data/processed/test",
        #transforms={"train": population_zscore_transform(0, 1)}
    )

    # Setup the DataModule
    datamodule.setup()

    # Get the DataLoader for training
    train_loader = datamodule.train_dataloader()

    # Fetch an example batch
    example_batch = next(iter(train_loader))

    # Inspect the batch
    print("Example Batch Shapes:")
    for i, tensor in enumerate(example_batch):
        # plot each tensor
        plt.plot(tensor[0].numpy().T)
        print(f"Tensor {i}: {tensor.shape}")


#%%

# %%
