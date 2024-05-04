# %%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# %%
class Scaler:
    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, array):
        self.mean = array.mean(dim=0)[None, None, ...].cuda(non_blocking=True)
        self.std = array.std(dim=0)[None, None, ...].cuda(non_blocking=True)

    def scale(self, array):
        return (array - self.mean) / self.std

    def unscale(self, array):
        return array * self.std + self.mean

    def unscale_without_mean(self, array):
        return array * self.std


def get_scalers(dataset):
    wtg0 = torch.tensor([])
    fct0 = torch.tensor([])
    output0 = torch.tensor([])
    for (wtg, fct), output in dataset:
        wtg0 = torch.concat((wtg0, wtg))
        fct0 = torch.concat((fct0, fct))
        output0 = torch.concat((output0, output))

    wtg_scaler = Scaler()
    wtg_scaler.fit(wtg0)
    fct_scaler = Scaler()
    fct_scaler.fit(fct0)
    output_scaler = Scaler()
    output_scaler.fit(output0)

    return wtg_scaler, fct_scaler, output_scaler


class LoaderScaler:
    def __init__(self, dataset) -> None:
        self.wtg_scaler, self.fct_scaler, self.output_scaler = get_scalers(
            dataset
        )

    def scale(self, input, target):
        return (
            (
                self.wtg_scaler.scale(input[0]),
                self.fct_scaler.scale(input[1]),
            ),
            self.output_scaler.scale(target),
        )

    def unscale(self, input, target):
        return (
            (
                self.wtg_scaler.unscale(input[0]),
                self.fct_scaler.unscale(input[1]),
            ),
            self.output_scaler.unscale(target),
        )

    def unscale_output(self, target):
        return self.output_scaler.unscale(target)

    def unscale_output_deviation(self, std):
        return self.output_scaler.unscale_without_mean(std)

    def unscale_output_variance(self, variance):
        return self.output_scaler.unscale_without_mean(
            self.output_scaler.unscale_without_mean(variance)
        )


# %%
class ExampleDataset(Dataset):
    """
    Dataset utilizado nos exemplos iniciais com MLP.
    Entrada: FCT grid LxDimx5x5. Saída Torre AMA Lx1
    """

    # load the dataset
    def __init__(self, tensor, data, length=7):
        # store the inputs and outputs
        self.tensor = tensor
        self.data = data.reset_index(drop=True)
        self.l = length

    # number of rows in the dataset
    def __len__(self):
        return len(self.data) - self.l - 1

    # get a row at an index
    def __getitem__(self, idx):
        A = self.tensor[idx : idx + self.l].flatten().cpu()
        B = torch.from_numpy(
            self.data[idx : idx + self.l]["wind_speed"].values
        ).cpu()

        return [A.float(), B.float()]

    def __iter__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)


# %%
class LSTM_FCT_WTG_Dataset_ws(Dataset):
    """
    Dataset utilizado na LSTM
    Entrada: FCT grid LxDimx5x5 + Dataframe[device_id, hour, features..., wind_speed]
    Argumentos opcionais referentes ao tamanho ddas entradas e das saídas
    """

    # load the dataset
    def __init__(
        self, tensor, fct_dts, data, L_past=24, L_future=24, debug=False
    ):
        indexes = []
        self.L_past = L_past
        self.L_future = L_future
        L = L_past + L_future
        self.L = L
        self.features = list(data.columns)[2:]

        length = pd.Timedelta(f"{str(L_past+L_future)} hour")
        for device_id, df in data.groupby("device_id"):
            print(f"Indexing for device {device_id}")
            for index, row in df[:-L].iterrows():
                if df.loc[index + L, "hour"] == row.hour + length:
                    indexes.append(index)
            if debug:
                break

        self.tensor = tensor.flatten(start_dim=1)
        self.fct_dts = fct_dts
        self.data = data
        self.indexes = indexes

    # number of rows in the dataset
    def __len__(self):
        return len(self.indexes)

    # get a row at an index
    def __getitem__(self, idx):
        # Separando L_past dados no passado e L_fut dados no futuro
        past_data = self.data[idx : idx + self.L_past]
        hour = past_data["hour"].dt.strftime("%Y-%m-%d %H:%M:%S").values[0]
        past_data = torch.from_numpy(past_data[self.features].values)
        fut_data = torch.from_numpy(
            self.data[idx + self.L_past : idx + self.L][["wind_speed"]].values
        )

        # Encontrando o index do horário que eu quero no Forecast
        idx2 = self.fct_dts[hour]

        # Nos dados do passado (Entrada), a feature de Forecast entra como variável
        # Cada instante t de past_data recebe FCT[t, t + L_future]
        past_data = torch.hstack(
            [
                past_data,
                torch.vstack(
                    [
                        self.tensor[
                            (idx2 + i) : (idx2 + i + self.L_future)
                        ].flatten()
                        for i in range(self.L_past)
                    ]
                ),
            ]
        )

        return [past_data.float(), fut_data.float()]

    def __iter__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)


# %%
class LSTM_FCT_WTG_Dataset_ws_xy(Dataset):
    """
    Dataset utilizado na LSTM
    Entrada: FCT grid LxDimx5x5 + Dataframe[device_id, hour, features..., wind_speed]
    Argumentos opcionais referentes ao tamanho ddas entradas e das saídas
    """

    # load the dataset
    def __init__(
        self, tensor, fct_dts, data, L_past=24, L_future=24, debug=False
    ):
        indexes = []
        self.L_past = L_past
        self.L_future = L_future
        L = L_past + L_future
        self.L = L
        self.features = list(data.columns)[2:]

        length = pd.Timedelta(f"{str(L_past+L_future)} hour")
        for device_id, df in data.groupby("device_id"):
            print(f"Indexing for device {device_id}")
            for index, row in df[:-L].iterrows():
                if df.loc[index + L, "hour"] == row.hour + length:
                    indexes.append(index)
            if debug:
                break

        self.tensor = tensor.flatten(start_dim=1)
        self.fct_dts = fct_dts
        self.data = data
        self.indexes = indexes

    # number of rows in the dataset
    def __len__(self):
        return len(self.indexes)

    # get a row at an index
    def __getitem__(self, idx):
        # Separando L_past dados no passado e L_fut dados no futuro
        past_data = self.data[idx : idx + self.L_past]
        hour = past_data["hour"].dt.strftime("%Y-%m-%d %H:%M:%S").values[0]
        past_data = torch.from_numpy(past_data[self.features].values)
        fut_data = torch.from_numpy(
            self.data[idx + self.L_past : idx + self.L][
                ["ws_x", "ws_y"]
            ].values
        )

        # Encontrando o index do horário que eu quero no Forecast
        idx2 = self.fct_dts[hour]

        # Nos dados do passado (Entrada), a feature de Forecast entra como variável
        # Cada instante t de past_data recebe FCT[t, t + L_future]
        past_data = torch.hstack(
            [
                past_data,
                torch.vstack(
                    [
                        self.tensor[
                            (idx2 + i) : (idx2 + i + self.L_future)
                        ].flatten()
                        for i in range(self.L_past)
                    ]
                ),
            ]
        )

        return [past_data.float(), fut_data.float()]

    def __iter__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)


# %%
class LSTM_FCT_WTG_Dataset_ws_xy_AB(Dataset):
    """
    Dataset utilizado na MLP
    Entrada: FCT grid LxDimx5x5 + Dataframe[device_id, hour, features..., wind_speed]
    Argumento TIPO DO DATASET:
        - Tipo A: Turbine-driven. Escolhe 4 turbinas pra ser o teste
        - Tipo B: Time-driven. Períodos de 2 dias. Escolhe 4 turbinas pra ser o teste
        - Tipo C: Particiona o ano em 2 pedaços
    Argumentos opcionais referentes ao tamanho ddas entradas e das saídas
    """

    # load the dataset
    def __init__(
        self,
        tensor,
        fct_dts,
        data,
        dataset_type,
        L_past=24,
        L_future=24,
        debug=False,
    ):
        indexes = []
        self.L_past = L_past
        self.L_future = L_future
        L = L_past + L_future
        self.L = L
        self.features = list(data.columns)[2:]
        self.dataset_type = dataset_type

        if "A_train" == dataset_type:
            data = data[
                ~data.device_id.isin([1103, 1106, 1303, 1310])
            ].reset_index(drop=True)
        if "A_test" == dataset_type:
            data = data[
                data.device_id.isin([1103, 1106, 1303, 1310])
            ].reset_index(drop=True)

        data.sort_values(["device_id", "hour"], ascending=True)

        length = pd.Timedelta(f"{str(L_past+L_future)} hour")
        for device_id, df in data.groupby("device_id"):
            print(f"Indexing for device {device_id}")
            for index, row in df[:-L].iterrows():
                if row.hour > pd.Timestamp("2020-12-30 00:00:00"):
                    continue
                hour_ = pd.to_datetime(row.hour)
                if not ((hour_.day % 2 == 1) and (hour_.hour == 0)):
                    continue
                if df.loc[index + L, "hour"] == row.hour + length:
                    if "C_train" == dataset_type:
                        if hour_ < pd.Timestamp("2020-09-01"):
                            indexes.append(index)
                    elif "C_test" == dataset_type:
                        if hour_ >= pd.Timestamp("2020-09-01"):
                            indexes.append(index)
                    else:
                        indexes.append(index)
            if debug:
                break

        self.tensor = tensor.flatten(start_dim=1)
        self.fct_dts = fct_dts
        self.data = data
        self.indexes = indexes

    def get_idx_hour(self, idx):
        idx = self.indexes[idx]
        past_data = self.data[idx : idx + self.L_past]
        hour = past_data["hour"].dt.strftime("%Y-%m-%d %H:%M:%S").values[0]

        return hour

    # number of rows in the dataset
    def __len__(self):
        return len(self.indexes)

    # get a row at an index
    def __getitem__(self, idx):
        # Separando L_past dados no passado e L_fut dados no futuro
        idx = self.indexes[idx]
        past_data = self.data[idx : idx + self.L_past]
        hour = past_data["hour"].dt.strftime("%Y-%m-%d %H:%M:%S").values[0]
        past_data = torch.from_numpy(past_data[self.features].values)
        fut_data = torch.from_numpy(
            self.data[idx + self.L_past : idx + self.L][
                ["ws_x", "ws_y"]
            ].values
        )

        # Encontrando o index do horário que eu quero no Forecast
        idx2 = self.fct_dts[hour]

        # Nos dados do passado (Entrada), a feature de Forecast entra como variável
        # Cada instante t de past_data recebe FCT[t, t + L_future]
        past_data = torch.hstack(
            [
                past_data,
                torch.vstack(
                    [
                        self.tensor[
                            (idx2 + i) : (idx2 + i + self.L_future)
                        ].flatten()
                        for i in range(self.L_past)
                    ]
                ),
            ]
        )

        return [past_data.float(), fut_data.float()]

    def __iter__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)


# %%
# %%
def verify(hour):
    if (hour >= pd.Timestamp("2021-03-01")) and (
        hour <= pd.Timestamp("2021-06-01")
    ):
        return "C_test"
    else:
        return "C_train"


# %%
class TwoInputDatasetGFS(Dataset):
    def __init__(
        self,
        tensor,
        fct_dts,
        data,
        dataset_type,
        L_past=24,  # 7 dias, 3-horário, D 00:00 a D+6  21:00
        L_future=24,  # 7 dias, 3-horário, D 00:00 a D+6  21:00
        debug=False,
    ):
        "Essa classe usa o Dataset GFS (0.GFSGetDataset.py)"
        indexes = []
        self.L_past = L_past
        self.L_future = L_future
        L = L_past + L_future
        self.L = L
        self.tensor = tensor
        self.features = list(data.columns)[2:]
        self.dataset_type = dataset_type

        if "A_train" == dataset_type:
            data = data[
                ~data.device_id.isin([1103, 1106, 1303, 1310])
            ].reset_index(drop=True)
        if "A_test" == dataset_type:
            data = data[
                data.device_id.isin([1103, 1106, 1303, 1310])
            ].reset_index(drop=True)

        if dataset_type == "App":
            hour = pd.Timestamp("2021-05-15 00:00:00")
            assert (
                hour - pd.Timestamp("2020-01-01")
            ).total_seconds() / 3600 % self.L == 0.0
            data = data[
                (data["hour"] >= hour)
                & (data["hour"] < hour + self.L * pd.Timedelta("1 hour"))
            ]

        data.sort_values(["device_id", "hour"], ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)

        length = pd.Timedelta(f"{str((self.L_past+self.L_future))} hour")

        for device_id, df in data.groupby("device_id"):
            print(f"Indexing for device {device_id}")
            aux = df[: -self.L] if dataset_type != "App" else df
            for index, row in aux.iterrows():
                hour_ = pd.to_datetime(row.hour)
                if not hour_.hour == 0:
                    continue
                if not (
                    (
                        int(
                            (
                                hour_ - pd.Timestamp("2020-01-01")
                            ).total_seconds()
                            / (3600)
                        )
                        % self.L
                        == 0
                    )
                ):
                    continue
                if index + self.L - 1 > aux.index.max():
                    continue
                if aux.loc[
                    index + self.L - 1, "hour"
                ] == row.hour + length - pd.Timedelta("1 hour"):
                    if (verify(hour_) == "C_train") and (
                        dataset_type == "C_train"
                    ):
                        print(device_id, hour_)
                        indexes.append(index)
                    elif (verify(hour_) == "C_test") and (
                        dataset_type == "C_test"
                    ):
                        indexes.append(index)
                    elif dataset_type[0] != "C":
                        indexes.append(index)
            if debug:
                break

        dic = {}
        i = 1
        for lista in fct_dts:
            for datetime in lista:
                dic[datetime] = i
                i += 1

        self.fct_dts = dic
        self.data = data
        self.indexes = indexes

        self.wtg_scaler = None
        self.fct_scaler = None
        self.output_scaler = None

    def get_idx_hour(self, idx):
        idx = self.indexes[idx]
        past_data = self.data[idx : idx + self.L_past]
        hour = past_data["hour"].dt.strftime("%Y-%m-%d %H:%M:%S").values[0]

        return hour

    # number of rows in the dataset
    def __len__(self):
        return len(self.indexes)

    # get a row at an index
    def __getitem__(self, index):
        idx = self.indexes[index]
        past_data = self.data[idx : idx + self.L_past]
        hour = past_data["hour"].values[0]
        interval = (hour - pd.Timestamp("2020-01-01")).round("3 H")
        past_data = torch.from_numpy(past_data[self.features].values)
        fut_data = torch.from_numpy(
            self.data[idx + self.L_past : idx + self.L][
                ["ws_x", "ws_y"]
            ].values
        )

        idx2 = int((interval.total_seconds()) / 3600) // 3

        fct_data = self.tensor[idx2 : (idx2 + self.L_past + self.L_future)]

        return [(past_data.float(), fct_data.float()), fut_data.float()]

    def __iter__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)


# %%
import random


class ToyDataset(Dataset):
    def __init__(
        self,
        data,
    ):
        self.data = data
        self.output = data**3 + random.gauss() / 10

    # number of rows in the dataset
    def __len__(self):
        return len(self.data)

    # get a row at an index
    def __getitem__(self, index):
        input = self.data[index]
        output = self.output[index]
        return torch.Tensor(input), torch.Tensor(output)

    def __iter__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)
