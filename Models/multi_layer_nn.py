from torch.nn import Module, Linear, Softmax
from torch.nn.functional import softmax, relu
import torch


class HiddenLayer(Module):
    def __init__(self, input_features):
        super(HiddenLayer, self).__init__()

        self.linear1 = Linear(input_features, 200)
        self.linear2 = Linear(200, 200)

    def forward(self, x):
        x = torch.add(x, -torch.mean(x))
        x = relu(self.linear1(x))

        x = torch.add(x, -torch.mean(x))
        x = relu(self.linear2(x))

        return x


class MultiLayerClassifier(Module):
    def __init__(self, input_features):
        super(MultiLayerClassifier, self).__init__()
        self.hidden = HiddenLayer(input_features=input_features)
        self.output = Linear(200, 10)

    def forward(self, x):
        hidden = self.hidden(x)
        confidence = softmax(self.output(hidden), dim=1)

        return hidden, confidence
