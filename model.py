import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from layers import ResidualBlock, CausalResidualBlock, UpsamplingDecoder, PositionalEncoding
from losses import CEtransitionLoss
from hydra.utils import instantiate


class SOD_v1(pl.LightningModule):
    def __init__(self, input_length, num_classes, num_filters1,
                num_filters2, num_filters3, learning_rate,
                nhead, num_attention_layers, dim_feedforward,
                dropout):

        super(SOD_v1, self).__init__()
        self.input_length = input_length
        self.num_classes = num_classes
        self.num_filters1 = num_filters1
        self.num_filters2 = num_filters2
        self.num_filters3 = num_filters3
        self.learning_rate = learning_rate
        self.nhead = nhead
        self.num_attention_layers = num_attention_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        
        # Convolutional front-end with residual blocks
        self.resblock1 = CausalResidualBlock(1, num_filters1, dilation=1)
        self.resblock2 = CausalResidualBlock(num_filters1, num_filters2, dilation=2)
        self.resblock3 = CausalResidualBlock(num_filters2, num_filters3, dilation=2)
        
        # Transformer setup
        d_model = num_filters3
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )

        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_attention_layers)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=200)
        
        self.decoder_lin = nn.Linear(d_model, num_classes)
        self.upsampler = UpsamplingDecoder(num_classes, num_classes, input_length)
        self.loss_fn = CEtransitionLoss()

    def forward(self, x):
        # x shape: (batch, 1, time)
        features = self.resblock1(x)
        features = self.resblock2(features)
        features = self.resblock3(features)
        
        # Transformer expects (seq_len, batch, d_model)
        features = features.permute(0, 2, 1)  # (time, batch, channels)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Pass through transformer
        transformed = self.transformer_encoder(features)
        
        # Decode to classes
        class_logits = self.decoder_lin(transformed)  # (batch, time_down, num_classes)
        class_logits = class_logits.permute(0, 2, 1)
        upsampled_logits = self.upsampler(class_logits)
        
        return upsampled_logits  # shape: (batch, num_classes, input_length)

    def compute_loss(self, logits, labels):
        return self.loss_fn(logits, labels)

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3, factor=0.5
            ),
            'monitor': 'val_loss' 
        }

        return [optimizer], [scheduler]
