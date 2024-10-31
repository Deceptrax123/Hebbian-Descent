from torch.nn import Module, Linear
from torch.nn.functional import softmax
import torch


class MultiLayerTest(Module):
    def __init__(self, input_features):
        super(MultiLayerTest, self).__init__()

        self.linear1 = Linear(input_features, 200)
        self.linear2 = Linear(200, 200)
        self.output = Linear(200, 10)

    def forward(self, x):
        x_centered = torch.add(x, -torch.mean(x))
        h1 = self.linear1(x_centered).relu()

        h_centered = torch.add(h1, -torch.mean(h1))
        h2 = self.linear2(h_centered).relu()

        h2_centered = torch.add(h2, -torch.mean(h2))
        logits = self.output(h2_centered)

        return softmax(logits, dim=1)
