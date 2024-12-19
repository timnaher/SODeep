#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.init import xavier_uniform_
from layers import CausalResidualBlock, PositionalEncoding, UpsamplingDecoder

class TransformerEncoderModel(nn.Module):
    def __init__(self, resblock_config, nhead, num_attention_layers, dim_feedforward, dropout, 
                 concat_intermediate=True, return_valid=True):
        super(TransformerEncoderModel, self).__init__()
        
        # Initialize the residual blocks
        self.residual_blocks = nn.ModuleList([
            CausalResidualBlock(
                in_filters=block['in_filters'],
                out_filters=block['out_filters'],
                dilation=block['dilation']
            ) for block in resblock_config
        ])
        
        # Determine the feature dimension based on `concat_intermediate`
        if concat_intermediate:
            self.d_model = sum(block['out_filters'] for block in resblock_config)
        else:
            self.d_model = resblock_config[-1]['out_filters']  # Only use the last block's output

        # Store the flags
        self.concat_intermediate = concat_intermediate
        self.return_valid = return_valid

        # Initialize Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_attention_layers)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=self.d_model, max_len=500)

    def forward(self, x, left_context=None):
        intermediate_outputs = []  # Store outputs from all residual blocks
        for block in self.residual_blocks:
            x = block(x)
            if self.concat_intermediate:
                intermediate_outputs.append(x)

        if self.concat_intermediate:
            # Concatenate along the channel dimension (dim=1)
            all_outs = torch.cat(intermediate_outputs, dim=1)  # (batch, combined_channels, time)
        else:
            # Only use the output of the last residual block
            all_outs = x  # (batch, last_block_channels, time)
        
        # Trim to valid time steps if `return_valid` is True (before Transformer)
        if self.return_valid:
            all_outs = all_outs[:, :, left_context:]  # (batch, channels, valid_time)
        # Permute for compatibility with positional encoding and transformer
        all_outs = all_outs.permute(0, 2, 1)  # (batch, time, features)
        all_outs = self.pos_encoder(all_outs)
        
        # Apply the Transformer
        all_outs = self.transformer(all_outs)

        return all_outs


# Define Decoder
class LinearDecoder(nn.Module):
    """Simple linear decoder with automatic upsampling length calculation."""
    def __init__(self, input_dim, intermediate_dim, num_classes, left_context=None):
        super(LinearDecoder, self).__init__()
        self.left_context = left_context
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, num_classes)
        )
      
    def forward(self, x, left_context=None):
        # Compute the upsampling length dynamically
        upsampling_length = x.size(2)  # Sequence length after cropping

        # Pass through the decoder
        x = self.decoder(x)
        x = x.permute(0, 2, 1)  # Adjust dimensions to BCT
        
        return self._crop_to_valid_length(x, left_context)

    def _crop_to_valid_length(self, x, left_context):
        if left_context is not None:
            x = x[:, :, left_context:]
        return x

class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes, dropout, left_context=None):
        super(LSTMDecoder, self).__init__()
        # Define the LSTM layer
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # add dropout
            dropout=dropout,
            batch_first=True
        )
        # Define the fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, left_context=None):
        if left_context is not None:
            x = x[:, left_context:, :] # input is in BTC

        # Pass input through LSTM
        out, _ = self.rnn(x)  # out: [batch_size, seq_len, hidden_dim]
        out = self.fc(out)  # out: [batch_size, seq_len, num_classes]
        out = out.permute(0, 2, 1) # Permutefor CrossEntropyLoss: [batch_size, num_classes, seq_len]
        return out


if __name__ == '__main__':
    # Define the model
    model = TransformerEncoderModel(
        resblock_config=[
            {'in_filters': 1, 'out_filters': 16, 'dilation': 1},
            {'in_filters': 16, 'out_filters': 16, 'dilation': 2},
            {'in_filters': 16, 'out_filters': 16, 'dilation': 4},
            {'in_filters': 16, 'out_filters': 32, 'dilation': 8},

        ],
        nhead=1,
        num_attention_layers=1,
        dim_feedforward=256,
        dropout=0.1
    )
    x = torch.randn(32, 1, 150)
    y = model(x)
    print('output of encoder:', y.shape)

    # Define the decoder
    decoder = LinearDecoder(
        input_dim=32,
        intermediate_dim=64,
        num_classes=3,
        upsampling_length=150
    )
    y = decoder(y)
    print(y.shape)

#%%