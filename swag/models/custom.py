# %%
import torch
from torch import nn

from .convlstm import ConvLSTM


# %%
class ConvLSTMFullModel(nn.Module):
    def __init__(
        self,
        num_classes,
        fct_input_size,
        wtg_input_size,
        hidden_size=1000,
        num_layers=2,
        seq_length=24,
        dropout_r=0.1,
    ):
        super(ConvLSTMFullModel, self).__init__()

        self.num_classes = num_classes
        self.fct_input_size = fct_input_size
        self.wtg_input_size = wtg_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        # convlstm layer
        self.convlstm = ConvLSTM(
            img_size=(5, 5),
            input_dim=fct_input_size * 2,  # devido a reshape
            hidden_dim=hidden_size,
            # num_layers=2,
            batch_first=True,
            kernel_size=(5, 5),
            return_sequence=True,
            bias=True,
            cnn_dropout=dropout_r,
            rnn_dropout=dropout_r,
        )

        # conv2d layer
        self.conv2d = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=(5, 5),
        )

        self.wtg_lstm = nn.LSTM(
            input_size=wtg_input_size,
            num_layers=1,
            hidden_size=hidden_size,
            bias=True,
            batch_first=True,
            # dropout=dropout_r,
        )
        self.dropout = nn.Dropout(p=dropout_r)

        self.sequential = nn.Sequential(
            nn.Linear(2 * seq_length * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_r),
            nn.Linear(hidden_size, 2 * seq_length),
        )

    def forward(self, x):
        (wtg, fct) = x

        # reshape
        fct = torch.concat(
            (
                fct[:, 0 : self.seq_length, :, :, :],
                fct[:, self.seq_length :, :, :, :],
            ),
            dim=2,
        )
        fct.shape

        # convlstm
        convlstm_out, _, _ = self.convlstm(fct)

        convlstm_out.shape

        # conv2d
        outs = []
        for t in range(convlstm_out.shape[1]):
            _out = self.conv2d(convlstm_out[:, t, :, :, :])[:, :, 0, 0]
            outs.append(_out[:, None, :])
        fct_out = torch.concat(outs, dim=1)
        fct_out.shape

        wtg_out, (_, _) = self.wtg_lstm(wtg)
        wtg_out = self.dropout(wtg_out)

        wtg_out.shape

        # flatten e dense final
        fct_out = fct_out.flatten(start_dim=1)
        wtg_out = wtg_out.flatten(start_dim=1)

        semifinal_out = torch.concat(
            (
                fct_out,
                wtg_out,
            ),
            dim=1,
        )
        semifinal_out.shape

        output = self.sequential(semifinal_out)
        output = output.reshape(output.shape[0], self.seq_length, 2)
        output.shape

        return output


# %%
class LSTMFullModel(nn.Module):
    def __init__(
        self,
        num_classes,
        fct_input_size,
        wtg_input_size,
        hidden_size=1000,
        num_layers=2,
        seq_length=24,
        dropout_r=0.0,
    ):
        super(LSTMFullModel, self).__init__()

        self.num_classes = num_classes
        self.fct_input_size = fct_input_size
        self.wtg_input_size = wtg_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.lstm = nn.LSTM(
            input_size=wtg_input_size
            + fct_input_size * 2 * 25,  # reshape + flatten
            num_layers=1,
            hidden_size=hidden_size,
            bias=True,
            batch_first=True,
            # dropout=dropout_r,
        )

        self.sequential = nn.Sequential(
            nn.Dropout(p=dropout_r),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_r),
            nn.Linear(hidden_size, 2 * seq_length),
            # nn.Linear(hidden_size * seq_length, hidden_size),
            # nn.Tanh(),
            # nn.Dropout(p=dropout_r),
            # nn.Linear(hidden_size, 2 * seq_length),
        )

    def forward(self, x):
        (wtg, fct) = x
        wtg.shape
        fct.shape

        # reshape
        fct = torch.concat(
            (
                fct[:, 0 : self.seq_length, :, :, :],
                fct[:, self.seq_length :, :, :, :],
            ),
            dim=2,
        )
        fct.shape

        # flatten
        fct = fct.flatten(start_dim=2)
        fct.shape

        fct_wtg = torch.concat((fct, wtg), dim=2)
        fct_wtg.shape

        _, (out_, _) = self.lstm(fct_wtg)  # (1 x Batche x Hidden)
        out_.shape

        out_ = out_.transpose(1, 0)  # (Batch x 1 x Hidden)
        out_.shape

        # flatten e dense final
        out_ = out_.flatten(start_dim=1)
        out_.shape

        output = self.sequential(out_)
        output = output.reshape(output.shape[0], self.seq_length, 2)
        output.shape

        return output


# %%
class MLPFullModel(nn.Module):
    def __init__(
        self,
        num_classes,
        fct_input_size,
        wtg_input_size,
        hidden_size=1000,  # ou 32
        num_layers=2,
        seq_length=24,
        dropout_r=0.1,
    ):
        super(MLPFullModel, self).__init__()

        self.num_classes = num_classes
        self.fct_input_size = (
            seq_length * fct_input_size * 25 * 2
        )  # reshape e flatten
        self.wtg_input_size = seq_length * wtg_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.mlp = nn.Sequential(
            nn.Linear(
                self.wtg_input_size + self.fct_input_size,
                hidden_size,
            ),
            nn.Tanh(),
            nn.Dropout(p=dropout_r),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_r),
            nn.Linear(hidden_size, seq_length * 2),
        )

    def forward(self, x):
        (wtg, fct) = x
        wtg.shape, fct.shape

        # reshape
        fct = torch.concat(
            (
                fct[:, 0 : self.seq_length, :, :, :],
                fct[:, self.seq_length :, :, :, :],
            ),
            dim=2,
        )
        fct.shape

        # flatten
        fct = fct.flatten(start_dim=1)
        wtg = wtg.flatten(start_dim=1)
        fct.shape, wtg.shape

        # concat in same tenor to mlp
        fct_wtg = torch.concat(
            (
                fct,
                wtg,
            ),
            dim=1,
        )
        fct_wtg.shape

        output = self.mlp(fct_wtg)
        output.shape

        output = output.reshape(output.shape[0], self.seq_length, 2)
        output.shape

        return output


# %%
class LSTMWTGOnly(nn.Module):
    def __init__(
        self,
        num_classes,
        fct_input_size,
        wtg_input_size,
        hidden_size=1000,
        num_layers=2,
        seq_length=24,
        dropout_r=0.0,
    ):
        super(LSTMWTGOnly, self).__init__()

        fct_input_size = 0
        self.num_classes = num_classes
        self.fct_input_size = 0
        self.wtg_input_size = wtg_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.lstm = nn.LSTM(
            input_size=wtg_input_size
            + fct_input_size * 2 * 25,  # reshape + flatten
            num_layers=1,
            hidden_size=hidden_size,
            bias=True,
            batch_first=True,
            # dropout=dropout_r,
        )

        self.sequential = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_r),
            nn.Linear(hidden_size, 2 * seq_length),
            # nn.Linear(hidden_size * seq_length, hidden_size),
            # nn.Tanh(),
            # nn.Dropout(p=dropout_r),
            # nn.Linear(hidden_size, 2 * seq_length),
        )

    def forward(self, x):
        (wtg, _) = x
        wtg.shape

        _, (out_, _) = self.lstm(wtg)  # (1 x Batche x Hidden)
        out_.shape

        out_ = out_.transpose(1, 0)  # (Batch x 1 x Hidden)
        out_.shape

        # flatten e dense final
        out_ = out_.flatten(start_dim=1)
        out_.shape

        output = self.sequential(out_)
        output = output.reshape(output.shape[0], self.seq_length, 2)
        output.shape

        return output


# %%
class ConvLSTMNWPOnly(nn.Module):
    def __init__(
        self,
        num_classes,
        fct_input_size,
        wtg_input_size,
        hidden_size=1000,
        num_layers=2,
        seq_length=24,
        dropout_r=0.1,
    ):
        super(ConvLSTMNWPOnly, self).__init__()

        self.num_classes = num_classes
        self.fct_input_size = fct_input_size
        wtg_input_size = 0
        self.wtg_input_size = 0
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        # convlstm layer
        self.convlstm = ConvLSTM(
            img_size=(5, 5),
            input_dim=fct_input_size * 2,  # devido a reshape
            hidden_dim=hidden_size,
            # num_layers=2,
            batch_first=True,
            kernel_size=(5, 5),
            return_sequence=True,
            bias=True,
            cnn_dropout=dropout_r,
            rnn_dropout=dropout_r,
        )

        # conv2d layer
        self.conv2d = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=(5, 5),
        )

        self.dropout = nn.Dropout(p=dropout_r)

        self.sequential = nn.Sequential(
            nn.Linear(seq_length * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_r),
            nn.Linear(hidden_size, 2 * seq_length),
        )

    def forward(self, x):
        (wtg, fct) = x

        # reshape
        fct = torch.concat(
            (
                fct[:, 0 : self.seq_length, :, :, :],
                fct[:, self.seq_length :, :, :, :],
            ),
            dim=2,
        )
        fct.shape

        # convlstm
        convlstm_out, _, _ = self.convlstm(fct)

        convlstm_out.shape

        # conv2d
        outs = []
        for t in range(convlstm_out.shape[1]):
            _out = self.conv2d(convlstm_out[:, t, :, :, :])[:, :, 0, 0]
            outs.append(_out[:, None, :])
        fct_out = torch.concat(outs, dim=1)
        fct_out.shape

        # flatten e dense final
        fct_out = fct_out.flatten(start_dim=1)

        output = self.sequential(fct_out)
        output = output.reshape(output.shape[0], self.seq_length, 2)
        output.shape

        return output


# %%
class GaussianBase(nn.Module):
    def __init__(self, base_class, *args, **kwargs):
        super(GaussianBase, self).__init__()

        self.mean = base_class(*args, **kwargs)
        self.variance = base_class(*args, **kwargs)
        self.soft = nn.Softplus()

    def forward(self, x):
        mean = self.mean(x)
        variance = self.soft(self.variance(x))

        return mean, variance


# %%
class DummyFullModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DummyFullModel, self).__init__()

    def forward(self, x):
        (wtg, _) = x

        return wtg[:, :, :2]
