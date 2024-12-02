# %%
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
print(hdf5_files)
# Open the large HDF5 file for appending data
with h5py.File(hdf5_file, 'a') as hdf:
    total_samples = 0  # Track the total number of samples across files

    for small_hdf5_file in hdf5_files:
        print(f"Processing {small_hdf5_file}")
        # Load data from the smaller HDF5 file
        with h5py.File(small_hdf5_file, 'r') as small_file:
            eeg_data = small_file['eeg'][:]
            labels = small_file['y'][:]

            # Assuming labels have as many columns as there are label elements
            for i, label_element in enumerate(label_elements):
                # Extract labels for the specific label element
                label_data = labels[:, i]
                num_new_samples = eeg_data.shape[0]

                # Resize datasets to accommodate new data
                hdf['data'].resize((total_samples + num_new_samples, eeg_data.shape[1]))
                hdf[f'labels{label_element}'].resize((total_samples + num_new_samples,))

                # Append new data and labels
                hdf['data'][total_samples:total_samples + num_new_samples] = eeg_data
                hdf[f'labels{label_element}'][total_samples:total_samples + num_new_samples] = label_data

            # Update total samples after processing one file
            total_samples += num_new_samples

# %%
# Load the HDF5 dataset to verify contents
with h5py.File(hdf5_file, 'r') as hdf:
    
    data = hdf['data'][:]
    labels_minus_5 = hdf['labels-5'][:]
    print("Keys in HDF5 file:", list(hdf.keys()))
    print(f"Shape of 'data': {data.shape}")
    print(f"Shape of 'labels-5': {labels_minus_5.shape}")

# %%
import h5py
import numpy as np

def balance_labels(hdf5_file):
    """
    Balances the labels in the HDF5 dataset by ensuring the same number of rows for each label.
    Assumes a single label dataset called 'label-5'.
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
            assert len(label_indices) >= min_count, (
                f"Not enough samples for label {label}. Found {len(label_indices)}, needed {min_count}."
            )
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

# File path to the HDF5 dataset
hdf5_file = 'processed/SO_data_fd.h5'

# Label elements to balance
label_elements = [-5]

# Balance the labels in the HDF5 dataset
balance_labels(hdf5_file)

# %%
