#%%
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf
import h5py
import numpy as np
from utils.transforms import population_zscore_transform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.signal import savgol_filter
import torch.nn.functional as F
import time
# Load the best model checkpoint for inference
def load_model_for_inference(checkpoint_path, model_config_path):
    # Load the model configuration (e.g., hydra configs)
    cfg = OmegaConf.load(model_config_path)
    
    # Instantiate the model
    model = instantiate(cfg.model)
    
    # Load the trained weights from the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

# File paths
basepath = "/Users/timnaher/Documents/PhD/Projects/SODeep/tb_logs/SOD_trans-lin/version_7"
checkpoint_path = basepath + "/checkpoints/" + [f for f in os.listdir(basepath + "/checkpoints") if f.endswith(".ckpt")][0]
model_config_path = basepath + "/hparams.yaml"
print(checkpoint_path)
print(model_config_path)

filename = "/Users/timnaher/Documents/PhD/Projects/SODeep/data/processed/test/sd_ses-7.h5"

# Load model
model = load_model_for_inference(checkpoint_path, model_config_path)
model.eval()

# Load the full-night data
with h5py.File(filename, "r") as f:
    data = f["eeg"][:]
    labels = f["targets"][:]

# z score the data with std 50 and mean 0
#data /= 20
# Define chunk size and strides
chunk_size = 150
stride     = 88

# Initialize empty list to store predictions
predictions = []
chunks = []
probas = []
#labels = []




# Loop through data in chunks
t1 = 1350000
t2 = 1450000
# Define the causal convolution's valid range in output
valid_start = 62
valid_end = 150
# Initialize empty lists to store non-overlapping data
valid_chunks = []
valid_predictions = []
valid_labels = []
valid_probas = []
inference_times = []

# Loop through data in chunks
# Loop through data in chunks
with torch.no_grad():
    for start_idx in range(t1, t2 - chunk_size + 1, stride):
        # Extract the chunk
        chunk = data[start_idx : start_idx + chunk_size].squeeze()  # Shape: (chunk_size,)

        # Add a batch and channel dimension (required by the model)
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, chunk_size)
        
        # Start timing
        start_time = time.time()

        # Run the model
        prediction = model(chunk_tensor)  # Prediction shape: [1, 3, valid_output_length]

        # Stop timing
        end_time = time.time()

        # Calculate and store inference time
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        # Compute the softmax
        probabilities = F.softmax(prediction / 0.1, dim=1)

        # Extract valid range of probabilities and smooth them
        probabilities_np = probabilities.squeeze(0).detach().numpy()  # Shape: [3, valid_length]
        smoothed_probabilities = savgol_filter(probabilities_np, window_length=3, polyorder=2, axis=1)

        # Predict classes for the valid range
        chunk_predictions = []
        for i in range(smoothed_probabilities.shape[1]):  # Loop through valid time points
            if smoothed_probabilities[0, i] < 0.20:  # Class 0 probability is less than 20%
                chunk_predictions.append(1 if smoothed_probabilities[1, i] > smoothed_probabilities[2, i] else 2)
            else:  # Default to class 0
                chunk_predictions.append(0)

        # Append only valid ranges to avoid duplication
        valid_chunks.extend(chunk[valid_start:])  # Append valid EEG data
        valid_predictions.extend(chunk_predictions)  # Append valid predictions
        valid_labels.extend(labels[start_idx + valid_start : start_idx + chunk_size])  # Append valid labels
        valid_probas.append(smoothed_probabilities)  # Append valid probabilities

# Concatenate valid probabilities along the time axis
probabilities = np.concatenate(valid_probas, axis=1)  # Shape: [3, total_valid_length]

# Convert valid lists to arrays
valid_chunks = np.array(valid_chunks)  # Shape: (total_valid_length,)
valid_predictions = np.array(valid_predictions)  # Shape: (total_valid_length,)
valid_labels = np.array(valid_labels).squeeze()  # Shape: (total_valid_length,)


#remove outliers from inference times
inference_times = np.array(inference_times) 
inference_times = inference_times[inference_times < 0.1]  # Remove outliers

# times 1000
inference_times *= 1000  # Convert to milliseconds
# plot histogram of inference times
plt.hist(inference_times, bins=20)
plt.xlabel("Inference Time (ms)")
plt.ylabel("Count")

# print statistics
print(f"Mean inference time: {inference_times.mean():.2f} ms")
print(f"Median inference time: {np.median(inference_times):.2f} ms")
print(f"Standard deviation of inference time: {inference_times.std():.2f} ms")

#%%
# Create a time axis based on the length of the data
time = np.arange(len(valid_chunks))


# Set the animation speed (slower interval)
animation_interval = 50  # in milliseconds

# Create the figure and subplots (3 subplots now)
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Function to update the animation
# Function to update the animation
def update(frame):
    # Clear previous lines
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()

    # Define window size
    window_size = 500  # Number of points visible at a time
    start = max(0, frame)
    end = min(len(time), frame + window_size)
    ymin1, ymax1 = -180, 150  # Y-axis limits for the EEG subplot
    ymin2, ymax2 = -0.5, 2.5  # Y-axis limits for the ground truth labels
    ymin3, ymax3 = -0.2, 1.2  # Y-axis limits for the softmax probabilities

    # First subplot: Softmax probabilities
    for i, label in zip(range(probabilities.shape[0]), ['baseline', 'UP->DOWN', 'DOWN->UP']):  # Loop over classes
        axs[0].plot(
            time[start:end], probabilities[i, start:end], label=fr"$P(\text{{{label}}})$",
            linewidth=2, alpha=0.95
        )
    axs[0].set_ylabel("Softmax Probabilities")
    axs[0].set_title("Softmax Probabilities Over Time")
    axs[0].legend(loc="upper left")
    axs[0].set_ylim(ymin3, ymax3)

    # Second subplot: EEG data with shaded predictions
    axs[1].plot(time[start:end], valid_chunks[start:end], label="EEG Data", color="black", alpha=0.8)
    axs[1].axhline(y=-75, color='r', linestyle='--', alpha=0.1)  # horizontal dashed line at y=-75
    axs[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)  # horizontal dashed line at y=0
    axs[1].fill_between(
        time[start:end], ymin1, ymax1,
        where=(np.array(valid_predictions[start:end]) == 1), color="green", alpha=0.3, label="UP->DOWN"
        )
    axs[1].fill_between(
        time[start:end], ymin1, ymax1,
        where=(np.array(valid_predictions[start:end]) == 2), color="red", alpha=0.3, label="DOWN->UP"
    )
    axs[1].set_ylabel("EEG")
    axs[1].set_title("EEG Data with Predictions")
    axs[1].legend(loc="upper left")
    axs[1].set_ylim(ymin1, ymax1)

    # Third subplot: Ground truth labels with shaded regions
    axs[2].plot(time[start:end], valid_labels[start:end], label="Ground Truth", color="blue", alpha=0.8)
    axs[2].fill_between(
        time[start:end], 0, 1,
        where=(np.array(valid_labels[start:end]) == 1), color="green", alpha=0.3, label="UP->DOWN"
        )
    axs[2].fill_between(
        time[start:end], 0, 1,
        where=(np.array(valid_labels[start:end]) == 2), color="red", alpha=0.3, label="DOWN->UP"
    )
    axs[2].set_ylabel("Labels")
    axs[2].set_title("Ground Truth Labels with Shaded Regions")
    axs[2].legend(loc="upper left")
    axs[2].set_ylim(0, 1)  # Shaded regions span from 0 to 1 on the y-axis

    # Set the x-axis limits for all subplots
    axs[0].set_xlim(time[start], time[end - 1])
    axs[1].set_xlim(time[start], time[end - 1])
    axs[2].set_xlim(time[start], time[end - 1])
    axs[2].set_xlabel("Time")

# Create the animation
ani = FuncAnimation(fig, update, frames=range(0, len(time), 10), interval=animation_interval, blit=False)

# Show the animation
plt.show()




