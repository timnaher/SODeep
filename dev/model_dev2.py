#%%


import torch
import torch.nn as nn
from typing import List, Sequence

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
    num_features=8,
    block_channels=[4] * 4,
    kernel_width=3,
    dialation=1,
)

causalencoder2 = TDSConvCausalEncoder(
    num_features=8,
    block_channels=[4] * 4,
    kernel_width=3,
    dialation=5,
)
x = torch.randn(128, 8, 150)
xhat = causalencoder1(x)
print(xhat.shape)  # Expected output: torch.Size([128, 16, 150])
xhat = causalencoder2(x)
print(xhat.shape)  # Expected output: torch.Size([128, 16, 150])




#%%
conv2dblock = TDSConv2dBlockCausal(
    channels=2, width=4,
    kernel_width=10, dilation=1,
    stride=2)
x = torch.randn(128, 8, 150)
xhat = conv2dblock(x)
print(xhat.shape)  # Expected output: torch.Size([128, 4, 150])






#%%
# INITIAL CONVOLUZIONAL BLOCK FOR CAUSAL ENCODER ##
class CausalConv1dBlock(nn.Module):
    """A causal 1D convolution block with layer normalization and dropout."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        norm_type: str = "layer", # can be "layer" or "batch"
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1),
        )
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(out_channels) if norm_type == "layer" else nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)




class CausalEncoder(nn.Module):
    """A multi-layer causal encoder with increasing dilation."""
    def __init__(
        self,
        in_channels: int, # input to the encoder
        conv_kernel_size: int,
        num_features: int, # output of the initial conv1d, must match input to TDSConvCausalEncoder
        block_channels: List[int],
        kernel_width: int,
        num_layers: int = 6,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            conv_kernel_size (int): Kernel size for the initial causal conv1d.
            num_features (int): Total number of features after the initial conv1d.
            block_channels (List[int]): List of block channel configurations for TDSConv2dBlockCausal.
            kernel_width (int): Kernel width for TDSConv2dBlockCausal.
            num_layers (int): Number of encoder layers.
        """
        super().__init__()
        self.initial_conv = CausalConv1dBlock(
            in_channels=in_channels,
            out_channels=num_features,
            kernel_size=conv_kernel_size,
        )

        # Create TDS blocks with increasing dilation
        layers = []
        for i in range(num_layers):
            dilation = 2**i  # Exponential increase in dilation
            layers.append(
                TDSConvCausalEncoder(
                    num_features=num_features,
                    block_channels=block_channels,
                    kernel_width=kernel_width,
                    dialation=dilation,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layers(x)
        return x


x = torch.randn(128, 4, 150)

net = CausalEncoder(
    in_channels=4,
    conv_kernel_size=3,
    num_features=16,
    block_channels=[8] * 4,
    kernel_width=3,
    num_layers=6,
)


class Decoder(nn.Module):
    """A causal decoder with increasing dilation."""
    def __init__(
        self,
        in_channels: int,
        num_features: int,
        block_channels: List[int],
        kernel_width: int,
        num_layers: int = 6,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            num_features (int): Total number of features after the initial conv1d.
            block_channels (List[int]): List of block channel configurations for TDSConv2dBlockCausal.
            kernel_width (int): Kernel width for TDSConv2dBlockCausal.
            num_layers (int): Number of decoder layers.
        """
        super().__init__()

        # Create TDS blocks with increasing dilation
        layers = []
        for i in range(num_layers):
            dilation = 2**i  # Exponential increase in dilation
            layers.append(
                TDSConvCausalDecoder(
                    num_features=num_features,
                    block_channels=block_channels,
                    kernel_width=kernel_width,
                    dialation=dilation,
                )
            )
        self.layers = nn.Sequential(*layers)

        self.final_conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.final_conv(x)
        return x
#%%