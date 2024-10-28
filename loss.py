import torch
import torch.nn.functional as F
from torch.nn import Module


class CrossEntropyLoss(Module):
    def __init__(self, weights=None, reduction=True):
        super(CrossEntropyLoss, self).__init__()
        self.weights = weights
        self.reduction = reduction

    def forward(self, predictions, targets):  # Predictions/Target:(B,C)

        liklihood = torch.log(F.softmax(predictions, dim=1))
        loss = -torch.sum(liklihood*targets, dim=1)

        if self.weights:
            loss = -torch.sum(self.weights*liklihood*targets, dim=1)  # (B,1)

        return torch.mean(loss, dim=0) if self.reduction else loss
