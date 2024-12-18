#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence


## ECODER Modules ##

## initial_conv1d
class CausalConv1dBlock(nn.Module):
    """A causal 1D convolution block with layer normalization and dropout.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for the 1D convolution.
        stride (int): Stride for the 1D convolution.
        norm_type (str): Type of normalization layer to use.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        norm_type: str = "layer",  # Can be "layer" or "batch"
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # No padding here; we handle it manually
        )
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(out_channels) if norm_type == "layer" else nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.kernel_size = kernel_size

    def forward(self, x):
        # Add left-side padding manually
        x = F.pad(x, (self.kernel_size - 1, 0))  # Padding (left, right)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)  # Swap for LayerNorm if needed
        x = self.dropout(x)
        return x

class TDSConv(nn.Module):
    """Time-depth separable 1D convolution block."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.1,
        norm_type: str = "layer",  # Can be "layer" or "batch"
    ):
        super().__init__()
        # Depthwise convolution (time-wise)
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,  # Same number of channels
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,  # Separate convolution for each channel
            padding=0,  # No padding here; we handle it manually
        )
        # Pointwise convolution (feature-wise)
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,  # Only across features
        )
        self.relu = nn.ReLU(inplace=True)
        self.norm_type = norm_type
        self.norm = nn.LayerNorm(out_channels) if norm_type == "layer" else nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Depthwise (time-wise) convolution with causal padding
        if self.depthwise_conv.kernel_size[0] > 1:
            x = F.pad(x, (self.depthwise_conv.kernel_size[0] - 1, 0))  # Left padding for causal
        x = self.depthwise_conv(x)
        x = self.relu(x)

        # Pointwise (feature-wise) convolution
        x = self.pointwise_conv(x)
        x = self.relu(x)

        # Apply normalization
        if self.norm_type == "layer":
            x = self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)  # LayerNorm requires last dim
        else:
            x = self.norm(x)  # BatchNorm1d works directly on the channel dimension

        # Apply dropout
        x = self.dropout(x)
        return x


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block."""
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.fc_block(inputs.swapaxes(-1, -2)).swapaxes(-1, -2) + inputs
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x





#%%
class Encoder(nn.Module):
    """A multi-layer causal encoder with increasing dilation.
    Args:
        in_channels (int): Number of input channels.
        conv_kernel_size (int): Kernel size for the initial causal conv1d.
        num_features (int): Total number of features after the initial conv1d.
        block_channels (List[int]): List of block channel configurations for TDSConv2dBlockCausal.
        kernel_width (int): Kernel width for TDSConv2dBlockCausal.
        num_layers (int): Number of encoder layers.
    """
    def __init__(
        self,
        in_channels: int, # input to the first initial conv1d
        init_conv_kernel_size: int, # kernel size for the initial conv1d
        init_num_features: int, # output of the initial conv1d, must match input to TDSConvCausalEncoder
    ):
        super().__init__()
        self.initial_conv = CausalConv1dBlock(
            in_channels=in_channels,
            out_channels=init_num_features,
            kernel_size=init_conv_kernel_size,
        )

        # Create TDS blocks with increasing dilation
        self.conv1 = TDSConv(in_channels=init_num_features, out_channels=64, kernel_size=3)
        self.fc1 = TDSFullyConnectedBlock(num_features=64)
        self.conv2 = TDSConv(in_channels=64, out_channels=64, kernel_size=3)
        self.fc2 = TDSFullyConnectedBlock(num_features=64)
        self.conv3 = TDSConv(in_channels=64, out_channels=64, kernel_size=3)
        self.fc3 = TDSFullyConnectedBlock(num_features=64)


    def forward(self, x):
        x = self.initial_conv(x)
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.conv2(x)
        x = self.fc2(x)
        x = self.conv3(x)
        x = self.fc3(x)
        return x


#%%

# DECODER Modules #

class MLPDecoder(nn.Module):
    """A 3-layer MLP decoder.
    Args:
        in_channels (int): Number of input channels from the encoder.
        hidden_channels (int): Number of hidden units in the MLP layers.
        num_classes (int): Number of output classes.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x):
        # Input: (batch, features, time)
        x = x.transpose(1, 2)  # Swap to (batch, time, features)
        x = self.mlp(x)        # Apply MLP to features at each time step
        x = x.transpose(1, 2)  # Swap back to (batch, num_classes, time)
        return x


# %% full net as a pytorch lightning module
import pytorch_lightning as pl
from losses import CEtransitionLoss


class TDS_1D_v1(pl.LightningModule):
    def __init__(self, input_length, num_classes,
                 learning_rate, nhead, num_attention_layers,
                 dim_feedforward, dropout, resblock_config,
                 pred_threshold, smoothness_weight=0.1,
                 transition_penalty_weight=0.1, lr_decay_nstep=4,
                 lr_decay_factor=0.5, intermediate_dim=128):
        """
        Args:
            resblock_config: List of dictionaries where each dict specifies
                             {'in_filters': int, 'out_filters': int, 'dilation': int}.
            intermediate_dim: Dimension of the intermediate layer in the decoder.
        """
        super(TDS_1D_v1, self).__init__()
        self.input_length = input_length
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.nhead = nhead
        self.num_attention_layers = num_attention_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.threshold = pred_threshold
        self.smoothness_weight = smoothness_weight
        self.transition_penalty_weight = transition_penalty_weight
        self.lr_decay_nstep = lr_decay_nstep
        self.lr_decay_factor = lr_decay_factor

        # Dynamically create residual blocks
        self.encoder = Encoder(in_channels=1, init_conv_kernel_size=3, init_num_features=128)
        self.decoder = MLPDecoder(in_channels=128, hidden_channels=intermediate_dim, num_classes=num_classes)

        self.loss_fn = CEtransitionLoss(
            self.smoothness_weight,
            self.transition_penalty_weight)

    def forward(self, x):
        # x shape: (batch, 1, time)
        x = self.encoder(x)
        logits = self.decoder(x)

        return logits # logits and probabilities
    
    def compute_loss(self, logits, labels):
        return self.loss_fn(
            logits, labels
        )

    def _common_step(self, batch, stage):
        data, labels = batch
        labels = labels.squeeze(-1)  # (batch, time)
        logits = self(data)

        # Prediction logic moved to a separate method
        preds = self._pred_from_logits(logits)

        loss = self.compute_loss(logits, labels)

        # Calculate accuracy
        acc = (preds == labels).float().mean()

        # Log overall metrics
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def _get_left_context(self) -> int:
        """
        Compute the receptive field (left context) based on CausalResidualBlocks.

        Each block contributes:
            left_context = (kernel_size - 1) * dilation
        """
        left, stride = 0, 1

        # Contribution from residual blocks
        for resblock in self.residual_blocks:
            kernel_size = resblock.kernel_size  # Fixed at 3 in your implementation
            dilation = resblock.conv1.dilation[0]  # Access dilation directly
            left += (kernel_size - 1) * dilation * stride
            stride *= 1  # Stride remains fixed as 1 in this model

        return left

    def _pred_from_logits(self, logits):
        """
        Converts probabilities to threshold-based predictions.
        """
        probabilities = F.softmax(logits, dim=1)
        preds = torch.zeros_like(probabilities[:, 0], dtype=torch.long)  # Default to class 0
        max_probs, max_classes = probabilities.max(dim=1)
        threshold_mask = max_probs >= 0.8
        preds[threshold_mask] = max_classes[threshold_mask]
        return preds


    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.lr_decay_nstep,
                gamma=self.lr_decay_factor
            ),
            'interval': 'epoch'  # Reduces learning rate every 3 epochs
        }

        return [optimizer], [scheduler]

# test it

# %%
