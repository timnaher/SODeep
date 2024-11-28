# %%
import h5py
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader

# Define HDF5 file path
hdf5_file = 'processed/SO_data_fd.h5'

# Define label elements to create corresponding datasets
label_elements = [-3, -5, -10]

# Initialize the HDF5 file and datasets
with h5py.File(hdf5_file, 'w') as hdf:
    # Create the data dataset
    data_ds = hdf.create_dataset(
        'data',
        shape=(0, 100),
        maxshape=(None, 100),  # Allow dynamic resizing in the first dimension
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

# %%
import numpy as np
import pandas as pd
import h5py

def process_csv(file_path, label_element):
    """Processes a CSV file and extracts data and a specific label element."""
    df = pd.read_csv(file_path)

    # Parse the 'data' column into arrays
    data_series = df['data'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',', dtype=np.float16))
    data_array = np.stack(data_series.values)
    
    # Parse the 'label_vector' column and extract the desired element as the label
    labels = df['label_vector'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',', dtype=np.int64)[label_element]).values
    
    return data_array, labels

# List of CSV files to process
csv_files = ["/Users/timnaher/Documents/PhD/Projects/SODeep/data/raw/random_array.csv"]

# Open the HDF5 file for appending data
with h5py.File(hdf5_file, 'a') as hdf:
    total_samples = 0  # Track the total number of samples across files
    
    for csv_file in csv_files:
        # Process each CSV file
        for label_element in label_elements:
            # Extract data and labels for the specific label element
            data_array, labels = process_csv(csv_file, label_element=label_element)
            num_new_samples = data_array.shape[0]

            # Resize datasets to accommodate new data
            hdf['data'].resize((total_samples + num_new_samples, 100))
            hdf[f'labels{label_element}'].resize((total_samples + num_new_samples,))

            # Append new data and labels
            hdf['data'][total_samples:total_samples + num_new_samples] = data_array
            hdf[f'labels{label_element}'][total_samples:total_samples + num_new_samples] = labels
        
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