#%%
import torch
from torch import nn
from typing import Literal, Sequence

class Permute(nn.Module):
    """Permute the dimensions of the input tensor."""
    def __init__(self, from_dims: str, to_dims: str) -> None:
        super().__init__()
        self._permute_idx = [from_dims.index(d) for d in to_dims]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._permute_idx)

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
        super().__init__()
        layers = {}
        layers["conv1d"] = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        layers["relu"] = nn.ReLU(inplace=True)
        layers["dropout"] = nn.Dropout(dropout)
        self.conv = nn.Sequential(*[layers[key] for key in layers])
        if norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, "norm"):
            x = self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x

class TDSConv2dBlockCausal(nn.Module):
    """A causal 2D temporal convolution block with channel x width splitting."""
    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
            stride=(1, 1),
            padding=(0, kernel_width - 1),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(channels * width)
        self.channels = channels
        self.width = width

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, C, T = inputs.shape
        x = inputs.reshape(B, self.channels, self.width, T)
        x = self.conv2d(x)
        x = x[..., :T]
        x = self.relu(x)
        x = x.reshape(B, C, T) + inputs
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
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

class TDSConvCausalEncoder(nn.Module):
    """A time depth-separable convolutional encoder."""
    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int],
        kernel_width: int,
    ) -> None:
        super().__init__()
        tds_conv_blocks = []
        for channels in block_channels:
            feature_width = num_features // channels
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlockCausal(channels, feature_width, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)

class TdsCausalStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_conv_kernel_width: int,
        in_conv_stride: int,
        num_blocks: int,
        channels: int,
        feature_width: int,
        kernel_width: int,
        out_channels: int | None = None,
    ):
        super().__init__()
        layers = {}
        C = channels * feature_width
        layers["conv1dblock"] = Conv1dBlock(
            in_channels,
            C,
            kernel_size=in_conv_kernel_width,
            stride=in_conv_stride,
        )
        layers["tds_block"] = TDSConvCausalEncoder(
            num_features=C,
            block_channels=[channels] * num_blocks,
            kernel_width=kernel_width,
        )
        self.layers = nn.Sequential(*layers.values())

        if out_channels is not None:
            self.linear_layer = nn.Linear(channels * feature_width, out_channels)

    def forward(self, x):
        x = self.layers(x)
        if hasattr(self, "linear_layer"):
            x = self.linear_layer(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x

class TDSCausalFeatureEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_kernel_width: int,
        tds_stage_config: list[dict],
        out_channels: int,
    ):
        super().__init__()
        self.initial_conv = Conv1dBlock(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=conv_kernel_width,
            stride=1,
            norm_type="layer",
            dropout=0.1,
        )
        tds_stages = [
            TdsCausalStage(**config) for config in tds_stage_config
        ]
        self.tds_stages = nn.Sequential(*tds_stages)
        self.projection = nn.Linear(64, out_channels)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.tds_stages(x)
        x = x.mean(dim=-1)
        x = self.projection(x)
        return x

x = torch.randn(16, 1, 150)

encoder = TDSCausalFeatureEncoder(
    in_channels=1,
    conv_kernel_width=3,
    tds_stage_config=[
        {"in_channels": 64, "num_blocks": 2, "channels": 8, "feature_width": 8, "kernel_width": 1, "in_conv_kernel_width": 3, "in_conv_stride": 1},
        {"in_channels": 64, "num_blocks": 2, "channels": 8, "feature_width": 8, "kernel_width": 1, "in_conv_kernel_width": 3, "in_conv_stride": 1},
        {"in_channels": 64, "num_blocks": 2, "channels": 8, "feature_width": 8, "kernel_width": 1, "in_conv_kernel_width": 3, "in_conv_stride": 1},
    ],
    out_channels=256,
)

output = encoder(x)
print(output.shape)  # Should print: torch.Size([16, 256])

#%%

import torch
import torch.nn as nn

class TDSConv2dBlockCausal(nn.Module):
    """A causal 2D temporal convolution block with channel x width splitting and dilation."""
    def __init__(self, channels: int, width: int, kernel_width: int, dilation: int = 1) -> None:
        """
        Args:
            channels (int): Number of input and output channels.
            width (int): Width of the feature dimension.
            kernel_width (int): Temporal kernel size.
            dilation (int): Dilation rate for the temporal convolution. Default is 1 (no dilation).
        """
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
            stride=(1, 1),
            padding=(0, (kernel_width - 1) * dilation),  # Adjust padding for dilation
            dilation=(1, dilation),  # Temporal dilation
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(channels * width)
        self.channels = channels
        self.width = width

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, C, T = inputs.shape

        # Reshape from (B, C, T) -> (B, channels, width, T)
        x = inputs.reshape(B, self.channels, self.width, T)

        # Apply causal 2D convolution
        x = self.conv2d(x)

        # Clip to ensure causality
        x = x[..., :T]

        # Apply ReLU activation
        x = self.relu(x)

        # Reshape back to (B, C, T) and add residual connection
        x = x.reshape(B, C, T) + inputs

        # Apply layer normalization
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x




class TDSConvCausalEncoder(nn.Module):
    """A time depth-separable convolutional encoder."""
    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int],
        kernel_width: int,
        dialation: int = 1,
    ) -> None:
        super().__init__()
        tds_conv_blocks = []
        for channels in block_channels:
            feature_width = num_features // channels
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlockCausal(channels, feature_width, kernel_width, dialation),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)


causalencoder1 = TDSConvCausalEncoder(
    num_features=16,
    block_channels=[4] * 4,
    kernel_width=3,
    dialation=1,
)

causalencoder2 = TDSConvCausalEncoder(
    num_features=16,
    block_channels=[4] * 4,
    kernel_width=3,
    dialation=5,
)
x = torch.randn(128, 16, 150)
xhat = causalencoder1(x)
print(xhat.shape)  # Expected output: torch.Size([128, 16, 150])
xhat = causalencoder2(x)
print(xhat.shape)  # Expected output: torch.Size([128, 16, 150])

# Idea:
# 1) first inital conv1d to make 32 channes out of 1 channel. should be causal, important for the rest of the network
# 2) then a series of TDSCausalFeatureEncoder blocks with increasing dilation

# %%
