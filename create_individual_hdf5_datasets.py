#%%
import h5py
import numpy as np
import os

if __name__ == "__main__":

    # Define directories
    input_dir = '/Volumes/T7/SchemAcS/labeled_windows_V1/'
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)

    # Get all .mat files in the input directory
    matfiles = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mat')]
    # delete files that contain '/.' in the
    matfiles = [f for f in matfiles if '/.' not in f]
    # Process each .mat file
    for i, matfile in enumerate(matfiles):
        try:
            # Load the .mat file
            with h5py.File(matfile, 'r') as src_file:
                # Extract data
                eeg = np.array(src_file['eeg_filt_causal'][:])  # Convert to NumPy array
                time = np.array(src_file['time'][:])
                indices = np.arange(eeg.shape[0])
                targets = np.array(src_file['targets'][:])

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
        except Exception as e:
            print(f"Error processing {matfile}: {e}")


    # if debug
    # Define the path to the HDF5 file
    hdf5_file = "data/processed/sd_ses-1.h5"

    # Open the HDF5 file
    with h5py.File(hdf5_file, "r") as hdf:
        # Load the EEG data
        eeg = hdf["eeg"][:]
        time = hdf["time"][:]
        targets = hdf["targets"][:]
        indices = hdf["indices"][:]


# %%
def check_unique_files(train_dir, val_dir, test_dir):
    train_files = set(Path(train_dir).glob("*.h5"))
    val_files = set(Path(val_dir).glob("*.h5"))
    test_files = set(Path(test_dir).glob("*.h5"))

    overlap = {
        "train_val": train_files & val_files,
        "train_test": train_files & test_files,
        "val_test": val_files & test_files,
    }

    for key, value in overlap.items():
        if value:
            print(f"Overlap detected in {key}: {value}")
        else:
            print(f"No overlap in {key}.")

check_unique_files("data/processed/train", "data/processed/val", "data/processed/test")



# %%