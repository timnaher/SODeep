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
basepath = "/Users/timnaher/Documents/PhD/Projects/SODeep/tb_logs/SOD_trans-lin/version_0"
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
t1 = 355000
t2 = 390000
# Define the causal convolution's valid range in output
valid_start = 62
valid_end = 150
# Initialize empty lists to store non-overlapping data
valid_chunks = []
valid_predictions = []
valid_labels = []
valid_probas = []
inference_times = []

# Initialize a rolling buffer
buffer = np.zeros(chunk_size, dtype=data.dtype)  # Rolling buffer for incoming data

# Pre-fill the buffer with the first `chunk_size` samples
buffer[:] = data[t1:t1 + chunk_size].squeeze()
# Initialize lists to store real-time predictions and corresponding chunks
real_time_predictions = []
real_time_probas = []
real_time_valid_chunks = []

# Real-time inference loop
with torch.no_grad():
    for idx in range(t1 + chunk_size, t2-100):
        # Slide the buffer to include the new sample
        buffer[:-1] = buffer[1:]  # Shift left
        buffer[-1] = data[idx]    # Add the new sample at the end

        # Convert buffer to tensor with required dimensions
        chunk_tensor = torch.tensor(buffer, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Run the model
        prediction = model(chunk_tensor)

        # Compute the softmax probabilities
        probabilities = F.softmax(prediction / 0.01, dim=1)
        probabilities_np = probabilities.squeeze(0).detach().numpy()  # Shape: [3, valid_length]

        # Smooth the probabilities
        smoothed_probabilities = savgol_filter(probabilities_np, window_length=3, polyorder=2, axis=1)

        # Extract valid range of probabilities and predictions
        valid_preds = []
        for i in range(smoothed_probabilities.shape[1]):
            if smoothed_probabilities[0, i] < 0.20:  # Class 0 probability is less than 20%
                valid_preds.append(1 if smoothed_probabilities[1, i] > smoothed_probabilities[2, i] else 2)
            else:
                valid_preds.append(0)
        valid_preds = np.array(valid_preds)

        # Append only new valid data
        real_time_predictions.extend([valid_preds[-1]])  # Append predictions for new valid range
        real_time_probas.append(smoothed_probabilities[:, -1])    # Append probabilities for new valid range
        real_time_valid_chunks.extend([buffer[-1]])  # Append corresponding EEG data

# Convert results to arrays
real_time_predictions = np.array(real_time_predictions)  # Shape: (total_valid_predictions,)
real_time_probas = np.array(real_time_probas)  # Shape: [3, total_valid_predictions]
real_time_valid_chunks = np.array(real_time_valid_chunks)  # Shape: (total_valid_predictions,)

#%%
# Define the range of interest
start = 3100
end = 3400

# Define the y-axis limits for the EEG data
ymin = real_time_valid_chunks[start:end].min() - 20
ymax = real_time_valid_chunks[start:end].max() + 20

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Plot the EEG data with shaded predictions
time_indices = np.arange(start, end)  # Correctly align time indices
axs[1].plot(time_indices, real_time_valid_chunks[start:end], label="EEG Data", color="black", alpha=0.8)
axs[1].axhline(y=-75, color='r', linestyle='--', alpha=0.1)  # Horizontal dashed line at y=-75
axs[1].fill_between(
    time_indices, ymin, ymax,
    where=(real_time_predictions[start:end] == 1), color="green", alpha=0.3, label="Predictions = 1"
)
axs[1].fill_between(
    time_indices, ymin, ymax,
    where=(real_time_predictions[start:end] == 2), color="blue", alpha=0.3, label="Predictions = 2"
)
axs[1].set_ylabel("EEG Data")
axs[1].set_title("EEG Data with Shaded Predictions")
axs[1].legend()

# Plot the probabilities
for i, label in enumerate(["Baseline", "UP->DOWN", "DOWN->UP"]):
    axs[0].plot(time_indices, real_time_probas[start:end,i ], label=f"P({label})", alpha=0.8)
axs[0].set_ylabel("Probabilities")
axs[0].set_title("Softmax Probabilities")
axs[0].legend()

# Plot the predictions
axs[2].plot(time_indices, real_time_predictions[start:end], label="Predictions", color="black", alpha=0.8)
axs[2].set_ylabel("Predictions")
axs[2].set_title("Predictions Over Time")
axs[2].legend()

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()

# %%
