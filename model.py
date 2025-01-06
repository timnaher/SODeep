#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from layers import ResidualBlock, CausalResidualBlock, UpsamplingDecoder, PositionalEncoding, SequentialLSTM
from losses import CEtransitionLoss
from hydra.utils import instantiate
from torchmetrics import Accuracy, Precision, Recall
from networks import TransformerEncoderModel, LinearDecoder, LSTMDecoder

# Base Model with Encoder and Decoder
class BaseModel(pl.LightningModule):
    def __init__(self, encoder, decoder, loss_fn, learning_rate, weight_decay,
                 return_valid, lr_scheduler_config=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.return_valid = return_valid
        self.lr_scheduler_config = lr_scheduler_config
        self.left_context = self._get_left_context()

        # if your encoder has a "return_valid" attribute
        self.loss_on_cropped = True if getattr(self.encoder, 'return_valid', False) or self.return_valid else False
        print(f"Left context: {self.left_context}")

    def forward(self, x):
        encoded = self.encoder(x, self.left_context)
        if self.return_valid:
            decoded = self.decoder(encoded, self.left_context)
        else:
            decoded = self.decoder(encoded)
        return decoded

    def compute_loss(self, logits, labels, **kwargs):
        return self.loss_fn(logits, labels, **kwargs)

    def _common_step(self, batch, stage):
        data, labels = batch
        labels = labels.squeeze(-1)

        if self.loss_on_cropped:
            labels = labels[:, self.left_context:]

        logits = self(data)
        preds = self._pred_from_logits(logits)
        loss = self.compute_loss(logits, labels, preds=preds)

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

    def _pred_from_logits(self, logits):
        probabilities = F.softmax(logits, dim=1)
        preds = torch.zeros_like(probabilities[:, 0], dtype=torch.long)
        max_probs, max_classes = probabilities.max(dim=1)
        threshold_mask = max_probs >= 0.8
        preds[threshold_mask] = max_classes[threshold_mask]
        return preds

    def _get_left_context(self) -> int:
        left_context = 0
        if hasattr(self.encoder, 'residual_blocks'):
            for block in self.encoder.residual_blocks:
                # Adjust based on your specific block definition
                kernel_size = block.kernel_size
                dilation = block.conv1.dilation[0]
                left_context += (kernel_size - 1) * dilation
        return left_context

    def configure_optimizers(self):
        # 1. Create an optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # 2. If config is provided, create a ReduceLROnPlateau
        if self.lr_scheduler_config:
            scheduler = {
                'scheduler': ReduceLROnPlateau(
                    optimizer,
                    mode=self.lr_scheduler_config.get('mode', 'min'),
                    factor=self.lr_scheduler_config.get('factor', 0.1),
                    patience=self.lr_scheduler_config.get('patience', 5),
                    verbose=self.lr_scheduler_config.get('verbose', False),
                    min_lr=self.lr_scheduler_config.get('min_lr', 1e-6),
                ),
                'monitor': 'val_loss',  # Must match the metric name you log
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [scheduler]

        # 3. If no lr_scheduler_config, return just the optimizer
        return [optimizer]



# Test the models
if __name__ == "__main__":
    # Encoder and decoder configuration
    resblock_config = [
        {'in_filters': 1, 'out_filters': 32, 'dilation': 2,},
        {'in_filters': 32, 'out_filters': 64, 'dilation': 3},
    ]
    encoder = TransformerEncoderModel(resblock_config, nhead=4, num_attention_layers=2, dim_feedforward=256, dropout=0.1)
    decoder = LSTMDecoder(input_dim=64, hidden_size=128, num_layers=2, num_classes=3)
    loss_fn = CEtransitionLoss(smoothness_weight=0.1, transition_penalty_weight=0.1)
    model = BaseModel(encoder, decoder, loss_fn, learning_rate=0.001, lr_scheduler_config={'step_size': 4, 'gamma': 0.5},return_valid=False)
    # run some random data
    data = torch.randn(2, 1, 150)
    y = model(data)
    print(y.shape)


#%%
'''
class SOD_v1(pl.LightningModule):
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
        super(SOD_v1, self).__init__()
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
        self.residual_blocks = nn.ModuleList()
        for block_config in resblock_config:
            self.residual_blocks.append(
                CausalResidualBlock(
                    in_filters=block_config['in_filters'],
                    out_filters=block_config['out_filters'],
                    dilation=block_config['dilation']
                )
            )

        # Transformer setup
        d_model = resblock_config[-1]['out_filters']  # Final number of filters
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )

        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_attention_layers)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=500)
        
        # Modify the decoder with two fully connected layers
        self.decoder_lin = nn.Sequential(
            nn.Linear(d_model, intermediate_dim),  # First fully connected layer
            nn.ReLU(),                             # Activation function
            nn.Linear(intermediate_dim, num_classes)  # Second fully connected layer
        )
        
        self.upsampler = UpsamplingDecoder(num_classes, num_classes, input_length)
        self.loss_fn = CEtransitionLoss(
            self.smoothness_weight,
            self.transition_penalty_weight)

    def forward(self, x):
        # x shape: (batch, 1, time)
        features = x
        for resblock in self.residual_blocks:
            features = resblock(features)

        # Transformer expects (seq_len, batch, d_model)
        features = features.permute(0, 2, 1)  # (batch, time, channels)

        # Add positional encoding
        features = self.pos_encoder(features)

        # Pass through transformer
        transformed = self.transformer_encoder(features)

        # Decode to classes
        class_logits = self.decoder_lin(transformed)  # (batch, time_down, num_classes)
        class_logits = class_logits.permute(0, 2, 1)
        upsampled_logits = self.upsampler(class_logits)

        return upsampled_logits # logits and probabilities
    
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

#%%
class SOD_lstm1(pl.LightningModule):
    def __init__(self, input_length, num_classes, num_filters1,
                 num_filters2, num_filters3, learning_rate,
                 hidden_size, num_lstm_layers, scale):
        super(SOD_lstm1, self).__init__()
        self.input_length = input_length
        self.num_classes = num_classes
        self.num_filters1 = num_filters1
        self.num_filters2 = num_filters2
        self.num_filters3 = num_filters3
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.scale = scale

        # Convolutional front-end with residual blocks
        self.resblock1 = CausalResidualBlock(1, num_filters1, dilation=2)
        self.resblock2 = CausalResidualBlock(num_filters1, num_filters2, dilation=3)
        self.resblock3 = CausalResidualBlock(num_filters2, num_filters3, dilation=4)
        self.resblock4 = CausalResidualBlock(num_filters2, num_filters3, dilation=5)
        self.resblock5 = CausalResidualBlock(num_filters2, num_filters3, dilation=6)

        # LSTM for sequential processing
        self.lstm = SequentialLSTM(
            in_channels=num_filters3,
            out_channels=num_classes * 2,  # Assuming pos and vel components
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            scale=scale
        )

        self.upsampler = UpsamplingDecoder(num_classes, num_classes, input_length)
        self.loss_fn = CEtransitionLoss()

    def forward(self, x):
        # x shape: (batch, 1, time)
        features = self.resblock1(x)
        features = self.resblock2(features)
        features = self.resblock3(features)
        features = self.resblock4(features)
        features = self.resblock5(features)
        
        # LSTM expects (batch, time, channels)
        features = features.permute(0, 2, 1)  # (batch, time, channels)
        
        # Reset LSTM state
        self.lstm.reset_state()

        # Process through LSTM
        outputs = []
        for t in range(features.shape[1]):  # Loop over time
            output = self.lstm(features[:, t])  # (batch, out_channels)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (batch, time, out_channels)
        
        # Decode to classes
        class_logits = outputs.permute(0, 2, 1)  # (batch, channels, time)
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

if __name__ == "__main__":
    model = SOD_v1(input_length=150, num_classes=3, num_filters1=32, num_filters2=64, num_filters3=128,
                   learning_rate=0.001, nhead=4, num_attention_layers=2, dim_feedforward=256, dropout=0.1)

    print(model)


#%%
'''