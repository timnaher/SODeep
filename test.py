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

import time
import torch
import torch.nn as nn
import torch.optim as optim

# Check device availability
device_cpu = torch.device("cpu")
device_mps = torch.device("mps") if torch.backends.mps.is_available() else None

# Simple model: A small MLP
class SimpleNet(nn.Module):
    def __init__(self, input_size=100, hidden_size=64, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Synthetic dataset
# Let's just create random inputs and targets
input_size = 1000
num_classes = 10
num_samples = 50000
batch_size = 1024

X = torch.randn(num_samples, input_size)
y = torch.randint(0, num_classes, (num_samples,))

# Simple data loader function
def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

def train_model(model, device, X, y, batch_size, epochs=2):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    start_time = time.time()
    for epoch in range(epochs):
        for xb, yb in get_batches(X, y, batch_size):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
    end_time = time.time()
    return end_time - start_time

# Train on CPU
model_cpu = SimpleNet(input_size=input_size, hidden_size=64, num_classes=num_classes)
cpu_time = train_model(model_cpu, device_cpu, X, y, batch_size)

print(f"CPU training time: {cpu_time:.4f} seconds")

if device_mps:
    # Train on MPS
    model_mps = SimpleNet(input_size=input_size, hidden_size=64, num_classes=num_classes)
    mps_time = train_model(model_mps, device_mps, X, y, batch_size)
    print(f"MPS training time: {mps_time:.4f} seconds")
    
    # Compare
    if mps_time < cpu_time:
        print("MPS was faster.")
    else:
        print("CPU was faster or roughly the same.")
else:
    print("MPS not available on this system.")


# %%
