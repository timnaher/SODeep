#%%
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import collections
from collections.abc import Sequence

from typing import Literal

import torch

from torch import nn


##################
# TDS FEATURIZER #
##################


class Permute(nn.Module):
    """Permute the dimensions of the input tensor.
    For example:
    ```
    Permute('NTC', 'NCT') == x.permute(0, 2, 1)
    ```
    """

    def __init__(self, from_dims: str, to_dims: str) -> None:
        super().__init__()
        assert len(from_dims) == len(
            to_dims
        ), "Same number of from- and to- dimensions should be specified for"

        if len(from_dims) not in {3, 4, 5, 6}:
            raise ValueError(
                "Only 3, 4, 5, and 6D tensors supported in Permute for now"
            )

        self.from_dims = from_dims
        self.to_dims = to_dims
        self._permute_idx: list[int] = [from_dims.index(d) for d in to_dims]

    def get_inverse_permute(self) -> "Permute":
        "Get the permute operation to get us back to the original dim order"
        return Permute(from_dims=self.to_dims, to_dims=self.from_dims)

    def __repr__(self):
        return f"Permute({self.from_dims!r} => {self.to_dims!r})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._permute_idx)


class BatchNorm1d(nn.Module):
    """Wrapper around nn.BatchNorm1d except in NTC format"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.permute_forward = Permute("NTC", "NCT")
        self.bn = nn.BatchNorm1d(*args, **kwargs)
        self.permute_back = Permute("NCT", "NTC")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.permute_back(self.bn(self.permute_forward(inputs)))


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_type: Literal["layer", "batch", "none"] = "layer",
        dropout: float = 0.0,
    ):
        """A 1D convolution with padding so the input and output lengths match."""

        super().__init__()

        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride

        layers = {}
        layers["conv1d"] = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

        if norm_type == "batch":
            layers["norm"] = BatchNorm1d(out_channels)

        layers["relu"] = nn.ReLU(inplace=True)
        layers["dropout"] = nn.Dropout(dropout)

        self.conv = nn.Sequential(
            *[layers[key] for key in layers if layers[key] is not None]
        )

        if norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type == "layer":
            x = self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()

        assert kernel_width % 2, "kernel_width must be odd."
        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
            dilation=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(channels * width)

        self.channels = channels
        self.width = width

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        B, C, T = inputs.shape  # BCT

        # BCT -> BcwT
        x = inputs.reshape(B, self.channels, self.width, T)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(B, C, -1)  # BcwT -> BCT

        # Skip connection after downsampling
        T_out = x.shape[-1]
        x = x + inputs[..., -T_out:]

        # Layer norm over C
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        x = inputs
        x = x.swapaxes(-1, -2)  # BCT -> BTC
        x = self.fc_block(x)
        x = x.swapaxes(-1, -2)  # BTC -> BCT
        x += inputs

        # Layer norm over C
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()
        self.kernel_width = kernel_width
        self.num_blocks = len(block_channels)

        assert len(block_channels) > 0
        tds_conv_blocks = []
        for channels in block_channels:
            feature_width = num_features // channels
            assert (
                num_features % channels == 0
            ), f"block_channels {channels} must evenly divide num_features {num_features}"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, feature_width, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class TdsStage(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        in_conv_kernel_width: int = 5,
        in_conv_stride: int = 1,
        num_blocks: int = 1,
        channels: int = 8,
        feature_width: int = 2,
        kernel_width: int = 1,
        out_channels: int | None = None,
    ):
        super().__init__()
        """Stage of several TdsBlocks preceded by a non-separable sub-sampling conv.

        The initial (and optionally sub-sampling) conv layer maps the number of
        input channels to the corresponding internal width used by the residual TDS
        blocks.

        Follows the multi-stage network construction from
        https://arxiv.org/abs/1904.02619.
        """

        layers: collections.OrderedDict[str, nn.Module] = collections.OrderedDict()

        C = channels * feature_width

        self.out_channels = out_channels

        # Conv1d block
        if in_conv_kernel_width > 0:
            layers["conv1dblock"] = Conv1dBlock(
                in_channels,
                C,
                kernel_size=in_conv_kernel_width,
                stride=in_conv_stride,
            )
        elif in_channels != C:
            # Check that in_channels is consistent with TDS
            # channels and feature width
            raise ValueError(
                f"in_channels ({in_channels}) must equal channels *"
                f" feature_width ({channels} * {feature_width}) if"
                " in_conv_kernel_width is not positive."
            )

        # TDS block
        layers["tds_block"] = TDSConvEncoder(
            num_features=C,
            block_channels=[channels] * num_blocks,
            kernel_width=kernel_width,
        )

        # Linear projection
        if out_channels is not None:
            self.linear_layer = nn.Linear(channels * feature_width, out_channels)

        self.layers = nn.Sequential(layers)

    def forward(self, x):
        x = self.layers(x)
        if self.out_channels is not None:
            x = self.linear_layer(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TdsNetwork(nn.Module):
    def __init__(
        self, conv_blocks: Sequence[Conv1dBlock], tds_stages: Sequence[TdsStage]
    ):
        super().__init__()
        self.layers = nn.Sequential(*conv_blocks, *tds_stages)
        self.left_context = self._get_left_context(conv_blocks, tds_stages)
        self.right_context = 0

    def forward(self, x):
        return self.layers(x)

    def _get_left_context(self, conv_blocks, tds_stages) -> int:
        left, stride = 0, 1

        for conv_block in conv_blocks:
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride

        for tds_stage in tds_stages:

            conv_block = tds_stage.layers.conv1dblock
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride

            tds_block = tds_stage.layers.tds_block
            for _ in range(tds_block.num_blocks):
                left += (tds_block.kernel_width - 1) * stride

        return left


#############
# NEUROPOSE #
#############


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        max_pool_size: tuple[int, int],
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        bn = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU()
        dropout = nn.Dropout(dropout_rate)
        maxpool = nn.MaxPool2d(kernel_size=max_pool_size, stride=max_pool_size)
        self.network = nn.Sequential(conv, bn, relu, dropout, maxpool)

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        num_convs: int,
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        def _conv(in_channels: int, out_channels: int):
            """Single convolution block."""
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]

        modules = [*_conv(in_channels, out_channels)]
        for _ in range(num_convs - 1):
            modules += _conv(out_channels, out_channels)

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.network(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        upsampling: tuple[int, int],
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        bn = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU()
        dropout = nn.Dropout(dropout_rate)
        scale_factor = (float(upsampling[0]), float(upsampling[1]))
        upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.network = nn.Sequential(conv, bn, relu, dropout, upsample)
        self.out_channels = out_channels

    def forward(self, x):
        return self.network(x)



############
# DECODERS #
############


class MLP(nn.Module):
    """Basic MLP with optional scaling of the final output."""

    def __init__(
        self,
        in_channels: int,
        layer_sizes: list[int],
        out_channels: int,
        layer_norm: bool = False,
        scale: float = 1.0,
    ):
        super().__init__()

        sizes = [in_channels] + layer_sizes
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            if layer_norm:
                layers.append(nn.LayerNorm(out_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(sizes[-1], out_channels))

        self.mlp = nn.Sequential(*layers)
        self.scale = scale

    def forward(self, x):
        # x is (batch, channel)
        return self.mlp(x) * self.scale




#%%
class TimeSeriesEncoder(nn.Module):
    """
    A time-series encoder leveraging Conv1d and TDSConvEncoder for temporal feature extraction.

    Args:
        input_channels (int): Number of input channels.
        conv_channels (list[int]): List of output channels for each Conv1dBlock.
        tds_channels (list[int]): List of output channels for each TDSConvEncoder block.
        kernel_size (int): Kernel size for Conv1d and TDS blocks.
        dropout (float): Dropout rate for Conv1d blocks.
    """
    def __init__(
        self,
        input_channels: int,
        conv_channels: list[int],
        tds_channels: list[int],
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Create initial Conv1d blocks
        self.conv_blocks = nn.ModuleList([
            Conv1dBlock(
                in_channels=input_channels if i == 0 else conv_channels[i - 1],
                out_channels=conv_channels[i],
                kernel_size=kernel_size,
                stride=1,
                norm_type="layer",
                dropout=dropout,
            )
            for i in range(len(conv_channels))
        ])

        # The final output size from Conv1d determines num_features
        num_features = conv_channels[-1]

        # Ensure num_features is divisible by all tds_channels
        for tds_channel in tds_channels:
            assert num_features % tds_channel == 0, (
                f"block_channels {tds_channel} must evenly divide num_features {num_features}"
            )

        # Temporal feature extraction with TDSConvEncoder
        self.tds_encoder = TDSConvEncoder(
            num_features=num_features,
            block_channels=tds_channels,
            kernel_width=kernel_size,
        )

    def forward(self, x):
        """
        Forward pass for the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time_steps).
        
        Returns:
            torch.Tensor: Encoded features.
        """
        # Ensure input is in NCT format (batch, channels, time)
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Apply TDSConvEncoder
        x = self.tds_encoder(x)

        return x



# %%

#%%
class TimeSeriesDecoder(nn.Module):
    """
    A time-series decoder that upsamples the encoded data back to the original input shape.

    Args:
        input_channels (int): Number of input channels for the decoder.
        conv_channels (list[int]): List of output channels for each ConvTranspose1d layer (in reverse order).
        output_channels (int): Number of output channels in the final layer (matching the encoder input channels).
        kernel_size (int): Kernel size for ConvTranspose1d layers.
        target_length (int): Desired temporal length for the reconstructed output.
        dropout (float): Dropout rate for ConvTranspose1d blocks.
    """
    def __init__(
        self,
        input_channels: int,
        conv_channels: list[int],
        output_channels: int,
        kernel_size: int = 5,
        target_length: int = 250,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Reverse the conv_channels for upsampling
        upsampling_channels = conv_channels[::-1]

        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=input_channels if i == 0 else upsampling_channels[i - 1],
                    out_channels=upsampling_channels[i],
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2  # Adjust padding to maintain size
                ),
                nn.BatchNorm1d(upsampling_channels[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            for i in range(len(upsampling_channels))
        ])

        # Final layer to map back to the original number of channels
        self.final_layer = nn.ConvTranspose1d(
            in_channels=upsampling_channels[-1],
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        # Optional upsampling layer to match the target temporal length
        self.upsample = nn.Upsample(size=target_length, mode='linear', align_corners=True)

    def forward(self, x):
        """
        Forward pass for the decoder.

        Args:
            x (torch.Tensor): Encoded tensor of shape (batch, channels, reduced_time_steps).

        Returns:
            torch.Tensor: Reconstructed tensor with the original input shape.
        """
        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.final_layer(x)

        # Upsample to the desired temporal length
        if self.upsample is not None:
            x = self.upsample(x)

        return x

# %%
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
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

        # Encoder-Decoder integration
        self.encoder = TimeSeriesEncoder(
            input_channels=1,        # Single input channel for time-series data
            conv_channels=[64, 128], # Convolutional layers for feature extraction
            tds_channels=[64, 128], # Temporal depth separable layers
            kernel_size=3,           # Kernel size for convolutions
            dropout=0.1              # Dropout rate
        )

        self.decoder = TimeSeriesDecoder(
            input_channels=128,      # Latent dimension from encoder output
            conv_channels=[128, 64], # Reverse architecture of encoder
            output_channels=3,       # Single output channel to match input
            kernel_size=3,           # Kernel size for transposed convolutions
            target_length=input_length, # Ensure output matches input length
            dropout=0.1              # Dropout rate
        )

        self.loss_fn = CEtransitionLoss(
            self.smoothness_weight,
            self.transition_penalty_weight
        )

    def forward(self, x):
        # x shape: (batch, 1, time)
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

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




# %%

