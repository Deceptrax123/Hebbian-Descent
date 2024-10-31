import torch
import torch.nn as nn
from torch.optim import Optimizer


class HebbianDescent(Optimizer):

    def __init__(self, params, lr):
        super(HebbianDescent, self).__init__(params, defaults={'lr': lr})

    def step(self, x, leaf_grad):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if len(p.data.size()) == 2:
                    p.data = p.data - group['lr']*torch.matmul(leaf_grad.t(),
                                                               torch.add(x, -torch.mean(x)))
                else:
                    p.data = p.data

    def zero_grad(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.zero_()
