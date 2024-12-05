#%%
import sys
sys.path.append('../')
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from utils.transforms import population_zscore_transform
from data.dataset import SOTimeSeriesDataset
from models.model import CNN_LSTM  

# Load configuration from YAML
config_path = "/Users/timnaher/Documents/PhD/Projects/SODeep/configs/models/CNN_LSTM.yaml"
cfg = OmegaConf.load(config_path)

def create_dataset(hdf5_file, z_score_mu, z_score_sigma, label_time):
    return SOTimeSeriesDataset(
        hdf5_file=hdf5_file,
        label_time=label_time,
        transform=population_zscore_transform(z_score_mu, z_score_sigma),
    )

train_data = create_dataset(
    cfg.data.train_hdf5_file, cfg.transforms.z_score_mu, cfg.transforms.z_score_sigma, label_time=-5
)
val_data = create_dataset(
    cfg.data.val_hdf5_file, cfg.transforms.z_score_mu, cfg.transforms.z_score_sigma, label_time=-5
)
test_data = create_dataset(
    cfg.data.test_hdf5_file, cfg.transforms.z_score_mu, cfg.transforms.z_score_sigma, label_time=-5
)

# Create DataLoaders
def create_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = create_dataloader(train_data, cfg.dataloader.batch_size, shuffle=True)
val_loader = create_dataloader(val_data, cfg.dataloader.batch_size, shuffle=False)
test_loader = create_dataloader(test_data, cfg.dataloader.batch_size, shuffle=False)

# Initialize the model
model = CNN_LSTM(cfg.model)

# Initialize logger
logger = TensorBoardLogger("tb_logs", name="CNN_LSTM_no_atten")

# Save hyperparameters in TensorBoard logger
logger.log_hyperparams(cfg)

# Callbacks for early stopping and checkpointing
early_stopping = EarlyStopping(
    monitor="val_loss",  # Metric to monitor
    mode="min",          # Stop when the monitored metric stops decreasing
    patience=5,          # Number of epochs to wait before stopping
    verbose=True,
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",   # Metric to monitor
    mode="min",           # Save the best model based on the lowest validation loss
    save_top_k=1,         # Save only the best model
    dirpath="checkpoints", # Directory to save checkpoints
    filename="best_model", # Name for the checkpoint file
    save_weights_only=True, # Save only the model weights
    verbose=True,
)

# Trainer setup
trainer = Trainer(
    max_epochs=cfg.training.max_epochs,
    logger=logger,
    log_every_n_steps=cfg.training.log_every_n_steps,
    devices="auto",
    accelerator="auto",
    callbacks=[early_stopping, checkpoint_callback],  # Add callbacks
)

# Train the model
trainer.fit(model, train_loader, val_loader)

# Test the best model (optional)
trainer.test(model, test_loader)

# %%
