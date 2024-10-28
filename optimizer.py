import torch
import torch.nn as nn
from torch.optim import Optimizer


class HebbianDescent(Optimizer):

    def __init__(self, params, lr):
        super(HebbianDescent, self).__init__(params, defaults={'lr': lr})

        self.state = dict()

    def step():
        pass
