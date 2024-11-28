#%%
import torch
from torch.utils.data import Dataset
import h5py

class SOTimeSeriesDataset(Dataset):
    def __init__(self, hdf5_file, label_time, transform=None):
        self.hdf5_file = hdf5_file
        self.label_time = label_time
        with h5py.File(self.hdf5_file, 'r') as hdf:
            self.length = hdf['data'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as hdf:
            data = hdf['data'][idx]
            label = hdf[f'labels{self.label_time}'][idx]
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return {'data': data, 'label': label}


#%% Test the dataset
dataset = SOTimeSeriesDataset(
    hdf5_file = 'SO_data_fd.h5',
    label_time = -3)


# get an element from the dataset
dataset[93]
# %%
