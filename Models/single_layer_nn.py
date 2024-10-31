from torch.nn import Linear, Module
from torch.nn.functional import softmax
import torch


class SingleLayerTest(Module):
    def __init__(self, input_features):
        super(SingleLayerTest, self).__init__()

        self.linear = Linear(in_features=input_features, out_features=10)

    def forward(self, x):
        h = self.linear(x)

        return h
