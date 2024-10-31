import torch
import torch.nn.functional as F
from torch.nn import Module


class CrossEntropyLoss(Module):
    def __init__(self, weights=None, reduction=True):
        super(CrossEntropyLoss, self).__init__()
        self.weights = weights
        self.reduction = reduction

    def forward(self, predictions, targets):  # Predictions/Target:(B,C)

        targets = torch.nn.functional.one_hot(targets, num_classes=10)
        log_liklihood = -1*torch.sum(targets*torch.log(predictions), dim=1)

        if self.weights is not None:
            log_liklihood = -1*torch.sum(
                targets*torch.log(predictions)*self.weights, dim=1)

        if self.reduction:
            return torch.mean(log_liklihood)
        else:
            return log_liklihood
