#%%
import torch.nn as nn
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from omegaconf import OmegaConf


class CNN_LSTM(pl.LightningModule):
    def __init__(self, cfg):
        super(CNN_LSTM, self).__init__()
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()

        # First Conv1d layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cfg.n_filters1, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=cfg.dropout1)

        # Second Conv1d layer
        self.conv2 = nn.Conv1d(in_channels=cfg.n_filters1, out_channels=cfg.n_filters2, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=cfg.dropout2)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=cfg.n_filters2, hidden_size=cfg.lstm_units, num_layers=cfg.lstm_layers,
                            batch_first=True, dropout=cfg.lstm_dropout)

        # Fully connected output layer
        self.fc = nn.Linear(cfg.lstm_units, cfg.num_classes)

    def forward(self, x):
        # Conv1D block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Conv1D block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # LSTM layer
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        x, (h_n, _) = self.lstm(x)

        # Fully connected layer
        x = self.fc(h_n[-1])  # Use the last hidden state for classification
        return x

    def _common_step(self, batch, stage: str):
        data, labels = batch
        outputs = self(data)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=-1) == labels).float().mean()

        # Log metrics based on stage
        if stage == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        elif stage == "val":
            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        elif stage == "test":
            self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate, weight_decay=0)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    # Define configuration using OmegaConf
    cfg = OmegaConf.create({
        "input_length": 150,
        "num_classes": 3,
        "n_filters1": 16,
        "n_filters2": 32,
        "dropout1": 0.3,
        "dropout2": 0.3,
        "lstm_units": 64,
        "lstm_layers": 2,
        "lstm_dropout": 0.1,
        "learning_rate": 0.01,
    })

    # Instantiate and test the model
    model = CNN_LSTM(cfg)
    x = torch.randn(32, 1, cfg.input_length)  # Batch of 32 samples
    output = model(x)
    print(output.shape)  # Expected output shape: (32, num_classes)

#%%