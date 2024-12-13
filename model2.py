#%%
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
from torch import nn
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import StepLR
from dataset import WindowedEEGDataModule


import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from layers import ResidualBlock, UpsamplingDecoder, PositionalEncoding

class SOD_v1(pl.LightningModule):
    def __init__(self, cfg):
        super(SOD_v1, self).__init__()
        self.cfg = cfg
        
        # Convolutional front-end with residual blocks
        self.initial_conv = nn.Conv1d(1, cfg.num_filters1, kernel_size=3, padding=1)
        
        self.resblock1 = ResidualBlock(cfg.num_filters1, cfg.num_filters1)
        self.resblock2 = ResidualBlock(cfg.num_filters1, cfg.num_filters2)
        self.resblock3 = ResidualBlock(cfg.num_filters2, cfg.num_filters3)
        
        # Optional downsampling
        #self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)
        
        d_model = cfg.num_filters3
        nhead = 4
        num_layers = 2
        dim_feedforward = 128
        dropout = 0.1
        
        # Define the transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=200)
        
        # Decoder linear layer before upsampling
        self.decoder_lin = nn.Linear(d_model, cfg.num_classes)
        self.upsampler = UpsamplingDecoder(cfg.num_classes, cfg.num_classes, cfg.input_length)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        # x shape: (batch, 1, time)
        features = self.initial_conv(x)
        features = self.resblock1(features)
        features = self.resblock2(features)
        features = self.resblock3(features)
        

        features = features.permute(2, 0, 1)  # (time, batch, channels)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Pass through transformer
        transformed = self.transformer_encoder(features)
        
        # Back to (batch, time, channels)
        transformed = transformed.permute(1, 0, 2)
        
        
        # Decode to classes
        class_logits = self.decoder_lin(transformed)  # (batch, time_down, num_classes)
        
        # Permute for upsampling
        class_logits = class_logits.permute(0, 2, 1)
        upsampled_logits = self.upsampler(class_logits)
        
        return upsampled_logits  # shape: (batch, num_classes, input_length)

    def compute_loss(self, logits, labels):
        # logits: (batch, num_classes, time)
        # labels: (batch, time)
        ce = self.ce_loss(logits, labels)

        # Smoothness penalty
        diff = (logits[:, :, 1:] - logits[:, :, :-1]).pow(2).mean()
        smoothness_penalty = 0.001 * diff

        # Transition penalty
        preds = logits.argmax(dim=1)  # Get predictions: (batch, time)

        # Define invalid transitions
        invalid_transitions = {
            (0, 2): 1,  # Penalize 0 → 2
            (2, 1): 1,  # Penalize 2 → 1
            (1, 0): 1,  # Penalize 1 → 0
        }

        penalties = []
        for batch_idx in range(preds.shape[0]):
            for t in range(preds.shape[1] - 1):
                current, next_ = preds[batch_idx, t], preds[batch_idx, t + 1]
                if (current.item(), next_.item()) in invalid_transitions:
                    penalties.append(invalid_transitions[(current.item(), next_.item())])

        if penalties:
            transition_penalty = 0.01 * torch.tensor(penalties, dtype=torch.float32).mean()
        else:
            transition_penalty = 0.0

        loss = ce + smoothness_penalty + transition_penalty
        return loss

    def _common_step(self, batch, stage):
        data, labels = batch
        labels = labels.squeeze(-1)  # (batch, time)
        
        logits = self(data)
        loss = self.compute_loss(logits, labels)
        
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
        return [optimizer], [scheduler]




# test the model
from omegaconf import OmegaConf
cfg = OmegaConf.create({

    "input_length": 150,
    "num_classes": 3,
    "num_filters1": 16,
    "num_filters2": 32,
    "num_filters3": 64,
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_workers": 0,
    "train_dir": "data/processed/train",
    "val_dir": "data/processed/val",
    "test_dir": "data/processed/test"
})

model =  SOD_v1(cfg)

# throw some random data in it
data = torch.randn(32, 1, 150)
out = model(data)
#print(out.shape)  # Expected: (32, 3, 150)

#%%

# DataModule
datamodule = WindowedEEGDataModule(
    window_length=cfg.input_length,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    train_dir=cfg.train_dir,
    val_dir=cfg.val_dir,
    test_dir=cfg.test_dir,
    stride =50,
    keep_ratio = 0,
    apply_savgol=True,
    deriv=1

)

# Model


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
    max_epochs=5,
    logger=logger,
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
)

# Train the model
trainer.fit(model, datamodule)



#%%
best_model_path = trainer.checkpoint_callback.best_model_path
model = EEGOscillationDetector.load_from_checkpoint(best_model_path, cfg=cfg)
model.eval()  # Set the model to evaluation mode

sample_input = torch.randn(32, 1, 150)  # Adjust to your input dimensions


import time

# Warm-up for GPU
if torch.cuda.is_available():
    model.to("cuda")
    _ = model(sample_input)

# Measure time
start_time = time.time()

# Perform inference
with torch.no_grad():
    for _ in range(100):  # Repeat to get average time
        _ = model(sample_input)

end_time = time.time()

avg_inference_time = (end_time - start_time) / 100  # Average time per inference
print(f"Average Inference Time: {avg_inference_time:.6f} seconds")

# %%
