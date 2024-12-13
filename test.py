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
checkpoint_path = "/Users/timnaher/Documents/PhD/Projects/SODeep/checkpoints/test-epoch=0-val_loss=0.17.ckpt"
model_config_path = "/Users/timnaher/Documents/PhD/Projects/SODeep/tb_logs/SOD_v1/version_3/hparams.yaml"
filename = "/Users/timnaher/Documents/PhD/Projects/SODeep/data/processed/test/sd_ses-20.h5"

# Load model
model = load_model_for_inference(checkpoint_path, model_config_path)

# Load the full-night data
with h5py.File(filename, "r") as f:
    data = f["eeg"][:]
    labels = f["targets"][:]


# z score the data with std 50 and mean 0
#data /= 20
# Define chunk size and stride
chunk_size = 150
stride = 150

# Initialize empty list to store predictions
predictions = []
chunks = []
model.eval()
# Loop through data in chunks
t1 = 76000
t2 = 85000
with torch.no_grad():
    for start_idx in range(t1,t2 - chunk_size + 1, stride):
        print(start_idx)
        # Extract the chunk
        chunk = data[start_idx : start_idx + chunk_size].squeeze()  # Shape: (chunk_size,)
        # plot the chunk 
        print(chunk.shape)
        chunks.append(chunk)
        
        #plt.plot(chunk)
        # generate random data as a chunk
        
        # Add a batch and channel dimension (required by the model)
        chunk = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, chunk_size)
        
        # Run the model
        prediction = model(chunk)
        
        # Convert prediction to NumPy array
        prediction_np = prediction.detach().numpy().squeeze()  # Shape: (num_classes, chunk_size)
        plt.plot(prediction_np.T)
        # Get predicted class for each time point in the chunk
        preds = np.argmax(prediction_np, axis=0)  # Shape: (chunk_size,)
        
        # Append predictions
        predictions.append(preds)
        

# Convert predictions to a NumPy array
predictions = np.concatenate(predictions)  # Combine all chunks' predictions into one array
all_chunks = np.concatenate(chunks)
print("Predictions shape:", predictions.shape)
sub_labels = np.squeeze(np.squeeze(labels[t1:t2]))
#%%
# Plot the predictions in 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
#plt.plot(labels[76000:85000], label="True labels")
axs[0].plot(predictions, label="Predictions", alpha=0.7,linestyle="--")
# title for axis 0
axs[0].set_title('Predictions')
#plt.plot(predictions, label="Predictions", alpha=0.7,linestyle="--")
axs[1].plot(labels[76000:85000], label="True labels")
# title for axis 1
axs[1].set_title('True labels')
#plt.plot(all_chunks, label="Data")
axs[2].plot(all_chunks)
# title for axis 2
axs[2].set_title('Data')
#plt.plot(all_chunks, label="Data")
plt.legend()

#%%
import matplotlib.pyplot as plt
import numpy as np

# Assuming predictions, labels, and all_chunks are defined
# predictions: array of 0, 1, 2 (categorical values)
# labels: true labels (used for the second subplot)
# all_chunks: EEG data

# Create a time axis based on the length of the data
# This represents the time indices corresponding to the EEG data points
time = np.arange(len(all_chunks))

# Create the figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# First subplot: EEG data with shaded predictions
# Plot the EEG signal as a black line with slight transparency for better visualization
axs[0].plot(time, all_chunks, label="EEG Data", color="black", alpha=0.8)

# Shade regions based on predictions
# For predictions == 1, shade the background green with partial transparency
axs[0].fill_between(
    time, all_chunks.min(), all_chunks.max(), 
    where=(predictions == 1), color="green", alpha=0.3, label="UP->DOWN"
)
# For predictions == 2, shade the background red with partial transparency
axs[0].fill_between(
    time, all_chunks.min(), all_chunks.max(), 
    where=(predictions == 2), color="red", alpha=0.3, label="DOWN->UP"
)

# Add labels, legend, and title to the first subplot
axs[0].set_ylabel("EEG Signal")
axs[0].set_title("EEG Data with Predictions")
axs[0].legend()

# Second subplot: Ground truth labels
# Plot the ground truth labels as a line
axs[1].plot(time, all_chunks, label="EEG Data", color="black", alpha=0.8)
axs[1].fill_between(
    time, all_chunks.min(), all_chunks.max(), 
    where=(sub_labels == 1), color="green", alpha=0.3, label="UP->DOWN"
)
# For predictions == 2, shade the background red with partial transparency
axs[1].fill_between(
    time, all_chunks.min(), all_chunks.max(), 
    where=(sub_labels == 2), color="red", alpha=0.3, label="DOWN->UP"
)
# Add labels, legend, and title to the second subplot
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Labels")
axs[1].set_title("Ground Truth Labels")
axs[1].legend()

# Set the x-axis limits for better focus
axs[0].set_xlim(1, 3000)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# Show the plot
plt.show()

# %%
model.eval()
cfg = OmegaConf.load(model_config_path)

datamodule = instantiate(cfg.datamodule)
datamodule.setup(stage="test")


#%% from here on it works!
# Generate Predictions on Test Data
model.eval()
test_dataloader = datamodule.test_dataloader()
predictions, ground_truths = [], []
with torch.no_grad():
    for batch in test_dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        predictions.append(outputs)
        ground_truths.append(targets)
        break  # Exit after processing one batch

predictions = torch.cat(predictions, dim=0)
ground_truths = torch.cat(ground_truths, dim=0)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(ground_truths.numpy(), label="Ground Truth", linestyle="--")
plt.plot(predictions.numpy(), label="Predictions", alpha=0.7)
plt.legend()
plt.title("Model Predictions vs Ground Truth (Single Batch)")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.show()
# %%
plt.plot(predictions.detach().numpy()[0,:,:].T)
plt.plot(ground_truths.T)
# %%


# %%
