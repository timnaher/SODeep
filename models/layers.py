#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

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

