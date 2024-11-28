import torch
from torch.utils.data import Dataset

class SOTimeSeriesDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, 'r') as hdf:
            self.length = hdf['data'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as hdf:
            data = hdf['data'][idx]
            label = hdf['labels'][idx]
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return {'data': data, 'label': label}