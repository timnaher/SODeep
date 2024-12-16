#%%
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf
import h5py
import numpy as np
from utils.transforms import population_zscore_transform
from model import SOD_v1  # Ensure this is your model definition
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.signal import savgol_filter
import torch.nn.functional as F

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
basepath = "/Users/timnaher/Documents/PhD/Projects/SODeep/tb_logs/SOD_v1/version_23"
checkpoint_path = basepath + "/checkpoints/" + [f for f in os.listdir(basepath + "/checkpoints") if f.endswith(".ckpt")][0]
model_config_path = basepath + "/hparams.yaml"
print(checkpoint_path)
print(model_config_path)

filename = "/Users/timnaher/Documents/PhD/Projects/SODeep/data/processed/test/sd_ses-18.h5"

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
stride     = 150

# Initialize empty list to store predictions
predictions = []
chunks = []
probas = []





# Loop through data in chunks
t1 = 81700
t2 = 120000
with torch.no_grad():
    for start_idx in range(t1, t2 - chunk_size + 1, stride):
        # Extract the chunk
        chunk = data[start_idx : start_idx + chunk_size].squeeze()  # Shape: (chunk_size,)
        chunks.append(chunk)
        
        # Add a batch and channel dimension (required by the model)
        chunk = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, chunk_size)
        
        # Run the model
        prediction, probabilities = model(chunk)  # `probabilities` shape: [1, 3, 150]

        # compute the softmax from predictions
        probabilities = F.softmax(prediction / 0.001, dim=1)


        
        # Extract probabilities (reshape from [1, 3, 150] to [3, 150])
        probabilities_np = probabilities.squeeze(0).detach().numpy()  # Shape: [3, 150]
        # Apply Savitzky-Golay filter to smooth the probabilities
        smoothed_probabilities = savgol_filter(probabilities_np, window_length=3, polyorder=2, axis=1)  # Shape: [3, 150]
        
        # Make predictions for the current chunk
        chunk_predictions = []
 
        # append the argmax of the smoothed probabilities
        for i in range(smoothed_probabilities.shape[1]):  # Loop through each time point in the chunk
            if smoothed_probabilities[0, i] < 0.20:  # Class 0 probability is less than 15%
                # Predict the class with the higher probability between class 1 and 2
                chunk_predictions.append(1 if smoothed_probabilities[1, i] > smoothed_probabilities[2, i] else 2)
            else:  # Default to class 0
                chunk_predictions.append(0)
        
        # Append predictions and smoothed probabilities
        predictions.append(chunk_predictions)  # Shape: (chunk_size,)
        probas.append(smoothed_probabilities)  # Shape: [3, chunk_size]

# Concatenate all chunks' predictions into one array
predictions = np.concatenate(predictions)  # Shape: (total_length,)

# Concatenate all probabilities along the time axis
probabilities = np.concatenate(probas, axis=1)  # Shape: [3, total_length]

# Concatenate all EEG chunks into one array
all_chunks = np.concatenate(chunks)  # Shape: (total_length,)

# Squeeze ground truth labels for the selected range
sub_labels = np.squeeze(np.squeeze(labels[t1:t2]))







#%%
# Create a time axis based on the length of the data
time = np.arange(len(all_chunks))


# Set the animation speed (slower interval)
animation_interval = 15  # in milliseconds

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
    for i,label in zip(range(probabilities.shape[0]),['baseline','UP->DOWN','DOWN->UP']):  # Loop over classes
        axs[0].plot(
            time[start:end], probabilities[i, start:end],  label=fr"$P(\text{{{label}}})$",
            linewidth=2, alpha=0.95
        )
    axs[0].set_ylabel("Softmax Probabilities")
    axs[0].set_title("Softmax Probabilities Over Time")
    axs[0].legend(loc="upper left")
    axs[0].set_ylim(ymin3, ymax3)

    # Second subplot: EEG data with shaded predictions
    axs[1].plot(time[start:end], all_chunks[start:end], label="EEG Data", color="black", alpha=0.8)
    axs[1].axhline(y=-75, color='r', linestyle='--', alpha=0.1)  # horizontal dashed line at y=-75
    axs[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)  # horizontal dashed line at y=0
    axs[1].fill_between(
        time[start:end], ymin1, ymax1,
        where=(predictions[start:end] == 1), color="green", alpha=0.3, label="UP->DOWN"
    )
    axs[1].fill_between(
        time[start:end], ymin1, ymax1,
        where=(predictions[start:end] == 2), color="red", alpha=0.3, label="DOWN->UP"
    )
    axs[1].set_ylabel("EEG")
    axs[1].set_title("EEG Data with Predictions")
    axs[1].legend(loc="upper left")
    axs[1].set_ylim(ymin1, ymax1)

    # Third subplot: Ground truth labels with shaded regions
    axs[2].plot(time[start:end], sub_labels[start:end], label="Ground Truth", color="blue", alpha=0.8)
    axs[2].fill_between(
        time[start:end], 0, 1,
        where=(sub_labels[start:end] == 1), color="green", alpha=0.3, label="UP->DOWN"
    )
    axs[2].fill_between(
        time[start:end], 0, 1,
        where=(sub_labels[start:end] == 2), color="red", alpha=0.3, label="DOWN->UP"
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
ani = FuncAnimation(fig, update, frames=range(0, len(time), 5), interval=animation_interval, blit=False)

# Show the animation
plt.show()




