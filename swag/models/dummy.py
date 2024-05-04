# ConvLSTM model from
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

__all__ = ["Dummy", "Dummy_CFG"]

import torch.nn as nn
import torch


class Dummy(nn.Module):

    # define model elements
    def __init__(self):
        super(Dummy, self).__init__()

    def __len__(self):
        return len(self.X)

    # forward propagate input
    def forward(self, X):
        return X[..., [0, 1]]


class Dummy_CFG:
    def __init__(self, base, *args, **kwargs):
        self.base = base
        self.args = list(args)
        self.kwargs = kwargs
