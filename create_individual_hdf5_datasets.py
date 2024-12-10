#%%
import h5py
import numpy as np
import os

# Define directories
input_dir = 'data/raw'
output_dir = 'data/processed'
os.makedirs(output_dir, exist_ok=True)

#%%

# Get all .mat files in the input directory
matfiles = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mat')]

# Process each .mat file
for i, matfile in enumerate(matfiles):
    # Load the .mat file
    with h5py.File(matfile, 'r') as src_file:
        # Extract data
        eeg = np.array(src_file['eeg'][:])  # Convert to NumPy array
        time = np.array(src_file['indices'][:])
        indices = np.arange(eeg.shape[0])
        targets = np.array(src_file['y'][:])

    # Define the output filename
    session_name = f"sd_ses-{i + 1}.h5"
    output_path = os.path.join(output_dir, session_name)

    # Save to HDF5 with the desired structure
    with h5py.File(output_path, 'w') as hdf_file:
        # Create the datasets
        hdf_file.create_dataset(
            'eeg',
            data=eeg,
            compression='gzip',
            chunks=True,  # Enable chunking for efficient access
            maxshape=(None,1)  # Allow dynamic resizing in the first dimension
        )
        hdf_file.create_dataset(
            'time',
            data=time,
            compression='gzip',
            chunks=True,  # Chunking for efficient access
            maxshape=(None)  # Allow dynamic resizing
        )
        hdf_file.create_dataset(
            'targets',
            data=targets,
            compression='gzip',
            chunks=True,  # Chunking for efficient access
            maxshape=(None,1)  # Allow dynamic resizing
        )
        hdf_file.create_dataset(
            'indices',
            data=indices,
            compression='gzip',
            chunks=True,  # Chunking for efficient access
            maxshape=(None,)  # Allow dynamic resizing
        )

    print(f"Processed {matfile} -> {output_path}")



# %%
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
        """
        windows = []
        zero_only_windows = []

        for start in range(0, self.num_timepoints - self.window_length + 1, self.stride):
            end = start + self.window_length
            windows.append((start, end))

            # Check if the window contains only zeros
            if (self.targets[start:end] == 0).all():
                zero_only_windows.append((start, end))

        # Randomly sample a subset of zero-only windows
        sampled_zero_windows = random.sample(zero_only_windows, int(len(zero_only_windows) * keep_ratio))

        # Combine sampled zero-only windows with all other windows
        non_zero_windows = [w for w in windows if w not in zero_only_windows]
        balanced_windows = non_zero_windows + sampled_zero_windows
        print(f"Balanced windows: {len(balanced_windows)}")
        print(f"Total windows: {len(windows)}")
        print(f"Zero-only windows: {len(zero_only_windows)}")
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


# test the dataset
# Define the path to the HDF5 file
hdf5_file = "data/processed/sd_ses-1.h5"

# Define the window length
window_length = 100

# Create the dataset
dataset = WindowedEEGDataset(hdf5_file, window_length)


# %%
