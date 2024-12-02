# %%
import h5py
import numpy as np
from omegaconf import OmegaConf

# Load configuration from YAML
config_path = "/Users/timnaher/Documents/PhD/Projects/SODeep/configs/data/data_config.yaml"
cfg = OmegaConf.load(config_path)


# Load the original dataset
with h5py.File(cfg.data.full_hdf5_file, 'r') as original_hdf:
    data = original_hdf['data'][:]
    num_samples = data.shape[0]
    labels_dict = {key: original_hdf[key][:] for key in original_hdf.keys() if key != 'data'}


indices = np.arange(num_samples)
np.random.shuffle(indices)

train_end = int(cfg.split.train_ratio * num_samples)
val_end = train_end + int(cfg.split.val_ratio* num_samples)

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

# Function to save split data into separate HDF5 files
def save_split_hdf5(file_path, data_split, label_splits):
    with h5py.File(file_path, 'w') as hdf:
        # Save data
        hdf.create_dataset(
            'data',
            data=data_split,
            dtype=np.float16,
            compression='gzip'
        )
        # Save labels
        for label_name, label_data in label_splits.items():
            hdf.create_dataset(
                label_name,
                data=label_data,
                dtype=np.int64,
                compression='gzip'
            )

# Save train dataset
save_split_hdf5(
    cfg.data.train_hdf5_file,
    data[train_indices],
    {key: labels[train_indices] for key, labels in labels_dict.items()}
)

# Save validation dataset
save_split_hdf5(
    cfg.data.val_hdf5_file,
    data[val_indices],
    {key: labels[val_indices] for key, labels in labels_dict.items()}
)

# Save test dataset
save_split_hdf5(
    cfg.data.test_hdf5_file,
    data[test_indices],
    {key: labels[test_indices] for key, labels in labels_dict.items()}
)

# %%
# Verify the separate HDF5 files

for file_path in [
    cfg.data.train_hdf5_file,
    cfg.data.val_hdf5_file,
    cfg.data.test_hdf5_file
    ]:
    with h5py.File(file_path, 'r') as hdf:
        print(f"Keys in {file_path}: {list(hdf.keys())}")
        for key in hdf.keys():
            print(f"{key} shape: {hdf[key].shape}")
# %%
