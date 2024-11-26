# / (Root Group)
# ├── data [Dataset of shape (N, 1000)]
# ├── labels [Dataset of shape (N,)]
# ├── experiment_ids [Dataset of shape (N,)]

#%%
import h5py
import numpy as np

hdf5_file = 'SO_data_fd.h5'

with h5py.File(hdf5_file, 'w') as hdf:
    data_ds = hdf.create_dataset(
        'data',
        shape=(0, 1000),
        maxshape=(None, 1000), # use None on the first dim to dynamically grow
        dtype=np.float32,
        chunks=True,
        compression='gzip'
    )
    labels_ds = hdf.create_dataset(
        'labels',
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64,
        chunks=True,
        compression='gzip'
    )
    exp_ids_ds = hdf.create_dataset(
        'experiment_ids',
        shape=(0,),
        maxshape=(None,),
        dtype='S10',  # String datatype
        chunks=True,
        compression='gzip'
    )
# %%
import os
import pandas as pd

def process_csv(file_path, experiment_id):
    df = pd.read_csv(file_path)
    data_series = df['data'].apply(lambda x: np.fromstring(x, sep=',', dtype=np.float32))
    data_array = np.stack(data_series.values)
    labels = df['label'].values.astype(np.int64)
    exp_ids = np.array([experiment_id.encode('utf8')] * len(labels))
    return data_array, labels, exp_ids

total_samples = 0

with h5py.File(hdf5_file, 'a') as hdf:
    data_ds = hdf['data']
    labels_ds = hdf['labels']
    exp_ids_ds = hdf['experiment_ids']

    for csv_file in csv_files:
        experiment_id = get_experiment_id_from_filename(csv_file)  # Implement this function
        data_array, labels, exp_ids = process_csv(csv_file, experiment_id)
        num_new_samples = data_array.shape[0]

        # Resize datasets
        data_ds.resize((total_samples + num_new_samples, 1000))
        labels_ds.resize((total_samples + num_new_samples,))
        exp_ids_ds.resize((total_samples + num_new_samples,))

        # Append data
        data_ds[total_samples:total_samples + num_new_samples] = data_array
        labels_ds[total_samples:total_samples + num_new_samples] = labels
        exp_ids_ds[total_samples:total_samples + num_new_samples] = exp_ids

        total_samples += num_new_samples

#%%




#%%
from torch.utils.data import DataLoader

dataset = TimeSeriesDataset('time_series_data.h5')
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

for batch in data_loader:
    data = batch['data']
    labels = batch['label']
    exp_ids = batch['experiment_id']
    # Training logic here
