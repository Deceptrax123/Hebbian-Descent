import torch
import torch.nn as nn
from torch.optim import Optimizer


class HebbianDescent(Optimizer):

    def __init__(self, params, lr):
        super(HebbianDescent, self).__init__(params, defaults={'lr': lr})

        self.state = dict()

    def step(self, x, leaf_grad):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data -= group['lr'] * \
                    torch.matmul(torch.add(x, -torch.mean(x)), leaf_grad)
