#%%
import h5py
import numpy as np
import os

# Define HDF5 file path
hdf5_file = 'processed/SO_data_fd.h5'

# Define label elements to create corresponding datasets
label_elements = [-5]

# Initialize the HDF5 file and datasets
with h5py.File(hdf5_file, 'w') as hdf:
    # Create the data dataset
    data_ds = hdf.create_dataset(
        'data',
        shape=(0, 150),
        maxshape=(None, 150),  # Allow dynamic resizing in the first dimension
        dtype=np.float32,
        chunks=True,
        compression='gzip'
    )
    # Create label datasets for each label element
    for label_element in label_elements:
        hdf.create_dataset(
            f'labels{label_element}',
            shape=(0,),
            maxshape=(None,),  # Allow dynamic resizing
            dtype=np.int64,
            chunks=True,
            compression='gzip'
        )

# List of smaller HDF5 files to process
# get all .mat files in /Volumes/T7/SchemAcS/labeled_windows/
# and store them in the list hdf5_files
hdf5_files = [os.path.join('/Volumes/T7/SchemAcS/labeled_windows/', f) for f in os.listdir('/Volumes/T7/SchemAcS/labeled_windows/') if f.endswith('.mat')]

train_files = hdf5_files[:int(len(hdf5_files)*0.8)]
val_files = hdf5_files[int(len(hdf5_files)*0.8):int(len(hdf5_files)*0.9)]
test_files = hdf5_files[int(len(hdf5_files)*0.9):]

#%%
import h5py
import os
import numpy as np

# Paths for the output HDF5 files
output_dir = "leave_one_par_out"
os.makedirs(output_dir, exist_ok=True)
train_file = os.path.join(output_dir, "train.h5")
val_file = os.path.join(output_dir, "val.h5")
test_file = os.path.join(output_dir, "test.h5")

# Initialize function to split and write data into separate HDF5 files
def create_hdf5_split(data_files, output_file, label_elements):
    with h5py.File(output_file, 'w') as hdf:
        # Create datasets with initial zero size
        data_ds = hdf.create_dataset(
            'data',
            shape=(0, 150),
            maxshape=(None, 150),
            dtype=np.float32,
            chunks=True,
            compression='gzip'
        )
        for label_element in label_elements:
            hdf.create_dataset(
                f'labels{label_element}',
                shape=(0,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=True,
                compression='gzip'
            )
        total_samples = 0
        for data_file in data_files:
            with h5py.File(data_file, 'r') as src_file:
                eeg_data = src_file['eeg'][:]
                labels = src_file['y'][:]
                num_new_samples = eeg_data.shape[0]

                # Resize datasets
                data_ds.resize((total_samples + num_new_samples, eeg_data.shape[1]))
                for i, label_element in enumerate(label_elements):
                    hdf[f'labels{label_element}'].resize((total_samples + num_new_samples,))

                # Append data and labels
                data_ds[total_samples:total_samples + num_new_samples] = eeg_data
                for i, label_element in enumerate(label_elements):
                    hdf[f'labels{label_element}'][total_samples:total_samples + num_new_samples] = labels[:, i]

                total_samples += num_new_samples

# Write train, validation, and test datasets
create_hdf5_split(train_files, train_file, label_elements)
create_hdf5_split(val_files, val_file, label_elements)
create_hdf5_split(test_files, test_file, label_elements)

# open one of the files to check the data
with h5py.File(train_file, 'r') as hdf:
    print(hdf.keys())
    print(hdf['data'].shape)
    print(hdf['labels-5'].shape)

# %%

# Paths to the datasets
train_file = 'leave_one_par_out/train.h5'
val_file = 'leave_one_par_out/val.h5'
test_file = 'leave_one_par_out/test.h5'

# Function to balance labels
def balance_labels(hdf5_file):
    """
    Balances the labels in the HDF5 dataset by ensuring the same number of rows for each label.
    Assumes a single label dataset called 'labels-5'.
    """
    with h5py.File(hdf5_file, 'a') as hdf:
        # Validate label dataset
        assert 'labels-5' in hdf, "Dataset 'labels-5' not found!"
        assert hdf['labels-5'].shape[0] == hdf['data'].shape[0], (
            f"Mismatch between 'labels-5' and 'data': "
            f"{hdf['labels-5'].shape[0]} != {hdf['data'].shape[0]}"
        )

        # Retrieve labels and data
        labels = hdf['labels-5'][:]
        data = hdf['data'][:]

        # Compute unique labels and their counts
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"Label distribution before balancing: {dict(zip(unique_labels, counts))}")
        min_count = np.min(counts)

        # Collect indices for balanced labels
        indices_to_keep = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            sampled_indices = np.random.choice(label_indices, size=min_count, replace=False)
            indices_to_keep.extend(sampled_indices)

        indices_to_keep = np.sort(indices_to_keep)  # Maintain order
        assert np.max(indices_to_keep) < data.shape[0], (
            f"Index out of bounds. Max index: {np.max(indices_to_keep)}, Dataset size: {data.shape[0]}"
        )

        # Resize datasets
        hdf['data'].resize((len(indices_to_keep), data.shape[1]))
        hdf['data'][:] = data[indices_to_keep, :]
        hdf['labels-5'].resize((len(indices_to_keep),))
        hdf['labels-5'][:] = labels[indices_to_keep]

        # Verify new label distribution
        new_labels = hdf['labels-5'][:]
        new_unique_labels, new_counts = np.unique(new_labels, return_counts=True)
        print(f"Label distribution after balancing: {dict(zip(new_unique_labels, new_counts))}")

# Balance each dataset
balance_labels(train_file)
balance_labels(val_file)
balance_labels(test_file)
# s
# %%
