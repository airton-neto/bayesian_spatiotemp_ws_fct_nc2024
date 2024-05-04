# %%
# Utilização
from forecast.get import load_data

tensor, fct_dts, data = load_data("Dataset")
# tensor: torch.tensor([TimeDim, VariableDim, LatDim, LonDim])
# fct_dts: dict[datetimes: str, datetime_index: int]
# data: pd.DataFrame(['device_id', 'hour', 'ws_x', 'ws_y', 'latitude', 'longitude'])

import torch

from forecast.dataset import TwoInputDatasetGFS
from forecast.get import load_loaders, save_loaders

# %% Testing a Dataset

dataset = TwoInputDatasetGFS(
    tensor, fct_dts, data, dataset_type="A_train", debug=True
)
loader = torch.utils.data.DataLoader(dataset, batch_size=1)

loaders = {"train": loader, "test": loader}

loaders = {
    "train": torch.utils.data.DataLoader(dataset, batch_size=128),
    "test": torch.utils.data.DataLoader(dataset, batch_size=1),
}

self = dataset

# %% Test a Model
from swag.models.custom import (
    ConvLSTMFullModel,
    DummyFullModel,
    GaussianBase,
    LSTMFullModel,
    MLPFullModel,
)

x, _ = next(iter(loader))
model_ = DummyFullModel(
    num_classes=2,
    fct_input_size=x[1].shape[2],
    wtg_input_size=x[0].shape[2],
)
self = model_
model_(x)

# %%
dataset_train_val = TwoInputDatasetGFS(
    tensor, fct_dts, data, dataset_type="A_train"
)
dataset_test = TwoInputDatasetGFS(tensor, fct_dts, data, dataset_type="A_test")

train_split = int(0.7 * len(dataset_train_val))
val_split = len(dataset_train_val) - train_split

dataset_train, dataset_val = torch.utils.data.random_split(
    dataset_train_val, [train_split, val_split]
)

loaders = {
    "train": torch.utils.data.DataLoader(dataset_train, batch_size=128),
    "val": torch.utils.data.DataLoader(dataset_val, batch_size=1),
    "train_val": torch.utils.data.DataLoader(
        dataset_train_val, batch_size=128
    ),
    "test": torch.utils.data.DataLoader(dataset_test, batch_size=1),
}

directory = save_loaders(
    loaders,
    dataset_train_val,
    dataset_train,
    dataset_val,
    dataset_test,
    "DatasetA",
)
_, _, _, _, _ = load_loaders(directory.split("/")[-1])


# %%
dataset = TwoInputDatasetGFS(tensor, fct_dts, data, dataset_type="B")

train_split = int(0.8 * 0.7 * len(dataset))
val_split = int(0.8 * 0.3 * len(dataset))
test_split = len(dataset) - train_split - val_split

dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
    dataset, [train_split, val_split, test_split]
)
dataset_train_val = torch.utils.data.ConcatDataset(
    (dataset_train, dataset_val)
)

loaders = {
    "train": torch.utils.data.DataLoader(dataset_train, batch_size=128),
    "val": torch.utils.data.DataLoader(dataset_val, batch_size=1),
    "train_val": torch.utils.data.DataLoader(
        dataset_train_val, batch_size=128
    ),
    "test": torch.utils.data.DataLoader(dataset_test, batch_size=1),
}

directory = save_loaders(
    loaders,
    dataset_train_val,
    dataset_train,
    dataset_val,
    dataset_test,
    "DatasetB",
)
_, _, _, _, _ = load_loaders(directory.split("/")[-1])


# %%
dataset_train_val = TwoInputDatasetGFS(
    tensor, fct_dts, data, dataset_type="C_train"
)
dataset_test = TwoInputDatasetGFS(tensor, fct_dts, data, dataset_type="C_test")

train_split = int(0.7 * len(dataset_train_val))
val_split = len(dataset_train_val) - train_split

dataset_train, dataset_val = torch.utils.data.random_split(
    dataset_train_val, [train_split, val_split]
)

loaders = {
    "train": torch.utils.data.DataLoader(dataset_train, batch_size=128),
    "val": torch.utils.data.DataLoader(dataset_val, batch_size=1),
    "train_val": torch.utils.data.DataLoader(
        dataset_train_val, batch_size=128
    ),
    "test": torch.utils.data.DataLoader(dataset_test, batch_size=1),
}

directory = save_loaders(
    loaders,
    dataset_train_val,
    dataset_train,
    dataset_val,
    dataset_test,
    "DatasetC",
)
_, _, _, _, _ = load_loaders(directory.split("/")[-1])

# %%
dataset_test = TwoInputDatasetGFS(
    tensor,
    fct_dts,
    data,
    dataset_type="App",
)

loaders = {
    "train": torch.utils.data.DataLoader(dataset_test, batch_size=1),
    "val": torch.utils.data.DataLoader(dataset_test, batch_size=1),
    "train_val": torch.utils.data.DataLoader(dataset_test, batch_size=1),
    "test": torch.utils.data.DataLoader(dataset_test, batch_size=1),
}

directory = save_loaders(
    loaders,
    dataset_test,
    dataset_test,
    dataset_test,
    dataset_test,
    "App",
)
_, _, _, _, _ = load_loaders(directory.split("/")[-1])

# %%
