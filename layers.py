#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TimeSeriesAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(TemporalAttention, self).__init__()
        self.attention_score = nn.Linear(input_dim, attention_dim)  # Transform features
        self.attention_context = nn.Linear(attention_dim, 1)       # Scalar attention score per time step

    def forward(self, features):
        """
        Args:
            features: Tensor of shape [batch_size, time_steps, feature_dim]
        Returns:
            attended_features: Tensor of shape [batch_size, time_steps, feature_dim]
            attention_weights: Attention weights [batch_size, time_steps]
        """
        attention_scores = self.attention_score(features)  # [batch_size, time_steps, attention_dim]
        attention_scores = F.elu(attention_scores)         # Apply non-linearity
        attention_weights = self.attention_context(attention_scores).squeeze(-1)  # [batch_size, time_steps]
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize over time for each time step
        
        # Compute attended features for each time step
        attended_features = features * attention_weights.unsqueeze(-1)  # Apply weights to features
        return attended_features, attention_weights


class UpsamplingDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, output_length):
        super(UpsamplingDecoder, self).__init__()
        # This is just a placeholder. Implement your upsampling logic here.
        # For demonstration, we won't do any complex operation:
        self.output_length = output_length

    def forward(self, x):
        # x: (batch, classes, time_down)
        # Implement your upsampling logic. 
        # For now, let's just return x as-is for this example.
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_filters, out_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_filters, out_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_filters)

        if in_filters != out_filters:
            self.downsample = nn.Conv1d(in_filters, out_filters, kernel_size=1)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        out = self.relu(out)
        return out



import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalResidualBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dilation=1):
        super(CausalResidualBlock, self).__init__()
        self.dilation = dilation
        self.kernel_size = 3
        self.padding = (self.kernel_size - 1) * dilation  # left padding only

        self.conv1 = nn.Conv1d(in_filters, out_filters, kernel_size=self.kernel_size, 
                               dilation=dilation, padding=0)
        self.bn1 = nn.BatchNorm1d(out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_filters, out_filters, kernel_size=self.kernel_size, 
                               dilation=dilation, padding=0)
        self.bn2 = nn.BatchNorm1d(out_filters)

        if in_filters != out_filters:
            self.downsample = nn.Conv1d(in_filters, out_filters, kernel_size=1)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        # Left-pad the input so that the convolution is causal
        # The padding tuple is (pad_left, pad_right)
        # We only pad on the left to ensure no future leakage.
        out = F.pad(x, (self.padding, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        # Create a positional encoding matrix for max_len positions
        # and d_model dimensions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension of size 1, since pe doesn't depend on batch
        pe = pe.unsqueeze(1)  # shape: (max_len, 1, d_model)
        
        # register_buffer so that it's not considered a parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to be of shape (seq_len, batch, d_model)
        seq_len = x.size(0)
        # Just add the positional encoding
        # pe[:seq_len] is of shape (seq_len, 1, d_model)
        # broadcasting will add it to x (seq_len, batch, d_model)
        return x + self.pe[:seq_len]