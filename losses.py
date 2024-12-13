import torch
import torch.nn as nn

class CEtransitionLoss(nn.Module):
    def __init__(self, smoothness_weight=0, transition_penalty_weight=0.1):
        super(CEtransitionLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.smoothness_weight = smoothness_weight
        self.transition_penalty_weight = transition_penalty_weight

    def forward(self, logits, labels):
        ce = self.ce_loss(logits, labels)


        # Smoothness penalty
        diff = (logits[:, :, 1:] - logits[:, :, :-1]).pow(2).mean()
        smoothness_penalty = self.smoothness_weight * diff

        # Transition penalty
        preds = logits.argmax(dim=1)  # Get predictions: (batch, time)
        invalid_transitions = {(0, 2): 1, (2, 1): 1, (1, 0): 1}
        penalties = []

        for batch_idx in range(preds.shape[0]):
            for t in range(preds.shape[1] - 1):
                current, next_ = preds[batch_idx, t], preds[batch_idx, t + 1]
                if (current.item(), next_.item()) in invalid_transitions:
                    penalties.append(invalid_transitions[(current.item(), next_.item())])

        if penalties:
            transition_penalty = self.transition_penalty_weight * torch.tensor(
                penalties, dtype=torch.float32, device=logits.device
            ).mean()
        else:
            transition_penalty = torch.tensor(0.0, device=logits.device)

        loss = ce + smoothness_penalty + transition_penalty
        return loss
