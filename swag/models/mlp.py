# ConvLSTM model from
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

__all__ = ["MLP", "GaussianMLP", "MLP_Dropout", "MLP_CFG"]

import torch.nn as nn


class MLP(nn.Module):

    # define model elements
    def __init__(self, n_inputs=1, n_outputs=1, n_hidden=32, dropout_p=0.1):
        super(MLP, self).__init__()
        n_inputs = n_inputs * 24
        n_outputs = n_outputs * 24
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(n_hidden, n_outputs),
        )

    # forward propagate input
    def forward(self, X):
        X = X.reshape([X.shape[0], X.shape[1] * X.shape[2]])
        X = self.layers(X)
        return X.reshape(X.shape[0], int(self.n_outputs / 2), 2)
        # return X


class GaussianMLP(nn.Module):

    # define model elements
    def __init__(self, n_inputs=1, n_outputs=1, n_hidden=32, dropout_p=0.1):
        super(GaussianMLP, self).__init__()
        n_inputs = n_inputs * 24
        n_outputs = n_outputs * 24
        self.n_outputs = n_outputs
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(n_hidden, n_outputs),
        )

        self.var_layers = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(n_hidden, n_outputs),
            nn.Softplus(),
        )

    # forward propagate input
    def forward(self, X):
        X = X.reshape([X.shape[0], X.shape[1] * X.shape[2]])
        Mean = self.layers(X)
        Var = self.var_layers(X)
        return (
            Mean.reshape(Mean.shape[0], int(self.n_outputs / 2), 2),
            Var.reshape(Var.shape[0], int(self.n_outputs / 2), 2),
        )


class MLP_Dropout(nn.Module):

    # define model elements
    def __init__(self, n_inputs=1, n_outputs=1, n_hidden=32, dropout_p=0.5):
        super(MLP_Dropout, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(n_hidden, n_outputs)
            # nn.Softplus()
        )

    # forward propagate input
    def forward(self, X):
        X = self.layers(X)
        return X

    def dropout(self, enable_bool):
        """Function to enable the dropout layers during test-time"""
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train(enable_bool)


class MLP_CFG:
    def __init__(self, base, *args, **kwargs):
        self.base = base
        self.args = list(args)
        self.kwargs = kwargs
