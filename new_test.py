#%%
import torch
from torch import nn
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import StepLR
from dataset import WindowedEEGDataModule


class EEGOscillationDetector(pl.LightningModule):
    def __init__(self, cfg):
        super(EEGOscillationDetector, self).__init__()
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()

        self.kernel_size = 3
        self.stride = 2
        self.padding = (self.kernel_size - 1) // 2

        # Convolutional layers with causal dilations
        self.causal1 = nn.Conv1d(1, cfg.num_filters, 3, stride=self.stride, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(cfg.num_filters)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)

        self.causal2 = nn.Conv1d(cfg.num_filters, cfg.num_filters, 3, stride=self.stride, padding=self.padding, dilation=2)
        self.bn2 = nn.BatchNorm1d(cfg.num_filters)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)

        self.causal3 = nn.Conv1d(cfg.num_filters, cfg.num_filters, 3, stride=self.stride, padding=self.padding, dilation=3)
        self.bn3 = nn.BatchNorm1d(cfg.num_filters)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)

        

        # Downsampled length
        self.downsampled_length = cfg.input_length
        for _ in range(2):
            self.downsampled_length = (self.downsampled_length - 1) // self.stride + 1

        # LSTM Decoder
        self.lstm = nn.LSTM(input_size=cfg.num_filters,
                            hidden_size=cfg.lstm_hidden_size,
                            num_layers=2,
                            batch_first=True)
        
        self.layer_norm = nn.LayerNorm(cfg.lstm_hidden_size)
        
        self.decoder = nn.Linear(cfg.lstm_hidden_size, cfg.num_classes)

        # Upsampling
        self.upsample = nn.Upsample(size=cfg.input_length, mode="linear", align_corners=False)

    def forward(self, x):
        features = self.causal1(x)
        features = self.bn1(features)
        features = self.relu1(features)
        features = self.dropout1(features)

        features = self.causal2(features)
        features = self.bn2(features)
        features = self.relu2(features)
        features = self.dropout2(features)

        features = self.causal3(features)
        features = self.bn3(features)
        features = self.relu3(features)
        features = self.dropout3(features)

        features = features.permute(0, 2, 1)
        lstm_out, _ = self.lstm(features)
        lstm_out = self.layer_norm(lstm_out)

        predictions = self.decoder(lstm_out)
        predictions = predictions.permute(0, 2, 1)
        predictions = self.upsample(predictions)
        return predictions

    def _common_step(self, batch, stage):
        data, labels = batch
        labels = labels.squeeze(-1)  # Remove the last dimension if it is 1
        outputs = self(data)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
        return [optimizer], [scheduler]


# Configuration
cfg = OmegaConf.create({
    "input_length": 150,
    "num_classes": 3,
    "num_filters": 64,
    "lstm_hidden_size": 256,
    "learning_rate": 0.01,
    "batch_size": 32,
    "num_workers": 0,
    "train_dir": "data/processed/train",
    "val_dir": "data/processed/val",
    "test_dir": "data/processed/test"
})

# DataModule
datamodule = WindowedEEGDataModule(
    window_length=cfg.input_length,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    train_dir=cfg.train_dir,
    val_dir=cfg.val_dir,
    test_dir=cfg.test_dir,
    stride =50,
    keep_ratio = 0.05
)

# Model
model = EEGOscillationDetector(cfg)

# Logger and Callbacks
logger = TensorBoardLogger("tb_logs", name="eeg_oscillation_detector")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints/",
    filename="oscillation-detector-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min",
)
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")

# Trainer
trainer = pl.Trainer(
    max_epochs=8,
    logger=logger,
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
)

# Train the model
trainer.fit(model, datamodule)


# Test the model
print("Testing the model")
trainer.test(model, datamodule)

print("Done training and testing the model")

#%%
import matplotlib.pyplot as plt
# Get the test dataloader
test_dataloader = datamodule.test_dataloader()

# Set the model to evaluation mode
model.eval()

# Get one batch of data
for batch in test_dataloader:
    data, labels = batch

    # Make predictions
    with torch.no_grad():
        predictions = model(data)

    # Convert predictions to class labels (argmax along the class dimension)
    pred_classes = predictions.argmax(dim=1)

    # Plot input, predictions, and ground truths for a few samples
    for i in range(min(20, len(data))):  # Plot up to 5 samples
        plt.figure(figsize=(6, 6))

        # Plot input signal
        plt.subplot(3, 1, 1)
        plt.plot(data[i].squeeze().cpu().numpy(), label="Input Signal")
        plt.title(f"Input Signal for Sample {i}")
        plt.legend()

        # Plot ground truth
        plt.subplot(3, 1, 2)
        plt.plot(labels[i].squeeze().cpu().numpy(), label="Ground Truth")
        plt.title(f"Ground Truth for Sample {i}")
        plt.legend()

        # Plot prediction
        plt.subplot(3, 1, 3)
        plt.plot(pred_classes[i].cpu().numpy(), label="Prediction")
        plt.title(f"Prediction for Sample {i}")
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Break after plotting the first batch
    break
# %%
