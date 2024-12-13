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
#checkpoint_path = "/Users/timnaher/Documents/PhD/Projects/SODeep/checkpoints/test-epoch=0-val_loss=0.17.ckpt"
#model_config_path = "/Users/timnaher/Documents/PhD/Projects/SODeep/tb_logs/SOD_v1/version_3/hparams.yaml"
basepath = "/Users/timnaher/Documents/PhD/Projects/SODeep/tb_logs/SOD_v1/version_12"
# take the only file in that is in basepath/checkpoints
checkpoint_path = basepath + "/checkpoints/" + [f for f in os.listdir(basepath + "/checkpoints") if f.endswith(".ckpt")][0]
model_config_path = basepath + "/hparams.yaml"
print(checkpoint_path)
print(model_config_path)

filename = "/Users/timnaher/Documents/PhD/Projects/SODeep/data/processed/test/sd_ses-20.h5"

# Load model
model = load_model_for_inference(checkpoint_path, model_config_path)

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
model.eval()
# Loop through data in chunks
t1 = 76000
t2 = 120000
with torch.no_grad():
    for start_idx in range(t1,t2 - chunk_size + 1, stride):
        # Extract the chunk
        chunk = data[start_idx : start_idx + chunk_size].squeeze()  # Shape: (chunk_size,)
        chunks.append(chunk)
        
        # Add a batch and channel dimension (required by the model)
        chunk = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, chunk_size)
        
        # Run the model
        prediction = model(chunk)
        
        # Convert prediction to NumPy array
        prediction_np = prediction.detach().numpy().squeeze()  # Shape: (num_classes, chunk_size)
        
        # Get predicted class for each time point in the chunk
        preds = np.argmax(prediction_np, axis=0)  # Shape: (chunk_size,)
        
        # Append predictions
        predictions.append(preds)

# Convert predictions to a NumPy array
predictions = np.concatenate(predictions)  # Combine all chunks' predictions into one array
all_chunks = np.concatenate(chunks)
print("Predictions shape:", predictions.shape)
sub_labels = np.squeeze(np.squeeze(labels[t1:t2]))



# Create a time axis based on the length of the data
time = np.arange(len(all_chunks))


# Set the animation speed (slower interval)
animation_interval = 15  # in milliseconds

# Create the figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Function to update the animation
def update(frame):
    # Clear previous lines
    axs[0].cla()
    axs[1].cla()

    # Define window size
    window_size = 500  # Number of points visible at a time
    start = max(0, frame)
    end = min(len(time), frame + window_size)
    ymin1, ymax1 = -150, 100  # Y-axis limits for the first subplot
    ymin2, ymax2 = -0.5, 2.5  # Y-axis limits for the second subplot

    # First subplot: EEG data with shaded predictions
    axs[0].plot(time[start:end], all_chunks[start:end], label="EEG Data", color="black", alpha=0.8)

    axs[0].fill_between(
        time[start:end], ymin1, ymax1,
        where=(predictions[start:end] == 1), color="green", alpha=0.3, label="UP->DOWN"
    )
    axs[0].fill_between(
        time[start:end], ymin1, ymax1,
        where=(predictions[start:end] == 2), color="red", alpha=0.3, label="DOWN->UP"
    )

    axs[0].set_ylabel("EEG ")
    axs[0].set_title("EEG Data with Predictions")
    axs[0].legend(loc="upper left")  # Fix the legend in the top-left corner
    axs[0].set_ylim(ymin1, ymax1)  # Fix y-axis for the first subplot

    # Second subplot: Ground truth labels
    axs[1].plot(time[start:end], sub_labels[start:end], label="Ground Truth", color="blue", alpha=0.8)

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Labels")
    axs[1].set_title("Ground Truth Labels")
    axs[1].legend(loc="upper left")  # Fix the legend in the top-left corner
    axs[1].set_ylim(ymin2, ymax2)  # Fix y-axis for the second subplot

    # Set the x-axis limits for both subplots
    axs[0].set_xlim(time[start], time[end - 1])
    axs[1].set_xlim(time[start], time[end - 1])

# Create the animation
ani = FuncAnimation(fig, update, frames=range(0, len(time), 5), interval=animation_interval, blit=False)

# Show the animation
plt.show()