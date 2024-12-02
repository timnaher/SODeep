#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(TemporalAttention, self).__init__()
        self.attention_score = nn.Linear(input_dim, attention_dim)  # Transform features
        self.attention_context = nn.Linear(attention_dim, 1)       # Scalar attention score

    def forward(self, features):
        """
        Args:
            features: Tensor of shape [batch_size, time_steps, feature_dim]
        Returns:
            context: Weighted feature vector [batch_size, feature_dim]
            attention_weights: Attention weights [batch_size, time_steps]
        """
        attention_scores = self.attention_score(features)  # [batch_size, time_steps, attention_dim]
        attention_scores = torch.tanh(attention_scores)    # Non-linearity
        attention_weights = self.attention_context(attention_scores).squeeze(-1)  # [batch_size, time_steps]
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize over time
        context = torch.bmm(attention_weights.unsqueeze(1), features).squeeze(1)  # Weighted sum
        return context, attention_weights

