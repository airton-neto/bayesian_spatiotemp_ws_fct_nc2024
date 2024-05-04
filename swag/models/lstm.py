__all__ = ["LSTM", "GaussianLSTM"]

import torch.nn as nn
from torch.autograd import Variable
import torch


class LSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        input_size,
        hidden_size,
        num_layers,
        seq_length,
        dropout=0,
    ):
        super(LSTM, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(
            hidden_size, num_classes
        )  # fully connected last layer

    def forward(self, x):
        h_0 = Variable(
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        )  # hidden state
        c_0 = Variable(
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        )  # internal state
        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(
            x, (h_0, c_0)
        )  # lstm with input, hidden, and internal state
        out = self.fc(out)  # Final Output
        return out


class GaussianLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        input_size,
        hidden_size,
        num_layers,
        seq_length,
        dropout=0,
    ):
        super(GaussianLSTM, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(
            hidden_size, num_classes
        )  # fully connected last layer

        self.lstm_var = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_var = nn.Linear(
            hidden_size, num_classes
        )  # fully connected last layer

        self.soft = nn.Softplus()

    def forward(self, x):

        h_0 = Variable(
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        )  # hidden state
        c_0 = Variable(
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        )  # internal state
        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(
            x, (h_0, c_0)
        )  # lstm with input, hidden, and internal state
        out = self.fc(out)  # Final Output

        # Propagate VAR
        out_var, (hn, cn) = self.lstm_var(
            x, (h_0, c_0)
        )  # lstm with input, hidden, and internal state
        out_var = self.fc_var(out_var)  # Final Out_varput
        out_var = self.soft(out_var)

        return out, out_var
