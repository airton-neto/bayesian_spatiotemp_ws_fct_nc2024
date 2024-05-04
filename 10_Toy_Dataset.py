# %% Intro
## Airton, em 2023-07-14.
## Projeto auto-contido para teste do código de extração de incerteza NLL

## requirements
# numpy
# torch
# tabulate
# properscoring
# matplotlib

# %%
import random
import warnings

import numpy as np
import properscoring as ps
import tabulate
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.functional import F
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class ToyDataset(Dataset):
    def __init__(
        self,
        data,
    ):
        self.data = data
        self.output = data**3 + np.random.normal(0, 1, size=data.size) / 8

    # number of rows in the dataset
    def __len__(self):
        return len(self.data)

    # get a row at an index
    def __getitem__(self, index):
        input = self.data[index]
        output = self.output[index]
        return [torch.Tensor([input]), torch.Tensor([output])]

    def __iter__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)


# %%
class ToyModel(nn.Module):
    def __init__(self, hidden_size=100, dropout=0.0, output=1):
        super(ToyModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(
                1,
                hidden_size,
            ),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(hidden_size, output),
        )

    def forward(self, x):
        return self.mlp(x)


class GaussianBase(nn.Module):
    def __init__(self, base_class, *args, **kwargs):
        super(GaussianBase, self).__init__()

        self.mean = base_class(*args, **kwargs, output=2)
        # self.variance = base_class(*args, **kwargs)
        self.soft = nn.Softplus()

    def forward(self, x):
        mean, variance = self.mean(x)[:, 0], self.mean(x)[:, 1]
        variance = self.soft(variance) + 1e-6

        return mean, variance


class GaussianMixture(nn.Module):
    def __init__(self, base_class, *args, **kwargs) -> None:
        super(GaussianMixture, self).__init__()
        for i in range(5):
            setattr(
                self,
                f"model_{str(i)}",
                GaussianBase(base_class=base_class, *args, **kwargs),
            )

    def forward(self, x):
        means = []
        variances = []
        for i in range(5):
            model = getattr(self, "model_" + str(i))
            mean, var = model(x)
            means.append(mean[..., None])
            variances.append(var[..., None])
        means = torch.stack(means, dim=2)
        mean = means.mean(dim=2)
        variances = torch.stack(variances, dim=2)
        variance = (variances + means.pow(2)).mean(dim=2) - mean.pow(2)
        return mean, variance


# %% Datasets
data_train = np.arange(-5, 5, 0.5)
data_test = np.arange(-7, 7, 0.5) + 0.1

scalemean = data_train.mean()
scalestd = data_train.std()

data_train = (data_train - scalemean) / scalestd
data_test = (data_test - scalemean) / scalestd

dataset_train = ToyDataset(data_train)
dataset_test = ToyDataset(data_test)

loaders = {
    "train": torch.utils.data.DataLoader(dataset_train, batch_size=1),
    "test": torch.utils.data.DataLoader(dataset_test, batch_size=1),
}

# %%
plt.plot(
    dataset_train.data.flatten(),
    dataset_train.output.flatten(),
    marker="o",
    linestyle="",
)
plt.plot(
    dataset_train.data.flatten(),
    (dataset_train.data**3).flatten(),
    marker="",
    linestyle="--",
)
plt.plot(
    dataset_test.data.flatten(),
    dataset_test.output.flatten(),
    marker="o",
    linestyle="",
)

# %% Testing model
x, _ = next(iter(loaders["train"]))

model = ToyModel()
model(x)

model = GaussianBase(base_class=ToyModel)
model(x)

model = GaussianMixture(base_class=ToyModel)
model(x)

# %% Definições
epochs = 3000
learning_rate = 0.001


def train_epoch(
    loader,
    model,
    criterion,
    optimizer,
):
    loss_sum = 0.0

    num_objects_current = 0

    model.train()

    for i, (input, target) in enumerate(loader):
        try:
            current_batch_size = input[0].size(0)
        except:
            current_batch_size = input.size(0)

        loss, output = criterion(model, input, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.data.item() * current_batch_size

        num_objects_current += current_batch_size

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None,
    }


def eval(
    loader,
    model,
    criterion,
):
    loss_sum = 0.0
    num_objects_current = 0

    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            try:
                current_batch_size = input[0].size(0)
            except:
                current_batch_size = input.size(0)

            loss, output = criterion(model, input, target)

            loss_sum += loss.item() * current_batch_size

            num_objects_current += current_batch_size

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None,
    }


def train_adversarial(*args, **kwargs):
    pass


def predict(
    loader, model, verbose=False, use_training_true=False, loader_scaler=None
):
    predictions = list()
    targets = list()
    if use_training_true:
        model.train()
    else:
        model.eval()

    with torch.no_grad():
        for input, target in loader:
            output = model(input)
            predictions.append(output)
            targets.append(target.numpy())

    return {
        "predictions": np.vstack(predictions),
        "targets": np.concatenate(targets),
    }


def predict_gaussian(loader, model, verbose=False, loader_scaler=None):
    predictions = list()
    variances = []
    targets = list()

    with torch.no_grad():
        for input, target in loader:
            output, variance = model(input)
            predictions.append(output)
            variances.append(variance)
            targets.append(target.numpy())

    return {
        "predictions": (np.vstack(predictions), np.vstack(variances)),
        "targets": np.concatenate(targets),
    }


# %% BayesianModelPredictor
def print_tabulate(values, epoch, scale=10):
    columns = ["Ep", "lr", "tr_loss", "te_loss", "swa_te_loss", "time"]
    table = tabulate.tabulate(
        [values], columns, tablefmt="simple", floatfmt="8.4f"
    )
    if epoch % (5 * scale) == 0:
        table = table.split("\n")
        if epoch % (50 * scale) == 0:
            table = "\n".join([table[1]] + table)
        else:
            table = table[2]
        print(table)


def get_metrics(Y, means, stds):
    # 1. RMSE
    rmse = np.sqrt(
        F.mse_loss(torch.from_numpy(means), torch.from_numpy(Y)).item()
    )

    # 2. Negative Log-Likelihood (Gaussian)
    nll = F.gaussian_nll_loss(
        torch.from_numpy(means), torch.from_numpy(Y), torch.from_numpy(stds)
    ).item()

    # 3. Continuous Ranked Probability Score
    crps = ps.crps_gaussian(Y, mu=means, sig=stds).mean()

    print("RMSE", rmse, "NLL", nll, "CRPS", crps)

    return rmse, nll, crps


def plot(dataset_train, dataset_test, Y, means, stds, label=""):
    rmse, nll, crps = get_metrics(Y, means, stds)

    data_test = dataset_test.data
    data_train = dataset_train.data
    plt.title(f"Model {label}\nRMSE {rmse:.4f} NLL {nll:.4f} CRPS {crps:4f}")
    plt.plot(
        data_test, Y, marker="o", linestyle="", markersize=2, color="blue"
    )
    plt.plot(
        data_test,
        data_test**3,
        marker="",
        markersize=2,
        label="real",
        color="blue",
    )
    plt.plot(
        data_train,
        dataset_train.output,
        marker="o",
        linestyle="",
        markersize=2,
        label="dados de treino",
        color="black",
    )
    plt.plot(data_test, means, marker="o", markersize=2, label="predito")
    plt.fill_between(
        data_test,
        (means - 3 * stds).flatten(),
        (means + 3 * stds).flatten(),
        color="#bdbdbd",
    )
    plt.legend()


def plot_loss(train_losses, test_losses):
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.legend()


class BayesianModelPredictor:
    def __init__(self, models, model_type):
        self.models = models
        self.model_type = model_type
        self.loader_scaler = None

    def _swag_sampler_method(self, model, loader, n_samples=25):
        model.sample(scale=10)
        init = predict(
            loader=loader, model=model, loader_scaler=self.loader_scaler
        )
        predictions = init["predictions"][..., None]
        Y = init["targets"]

        # Outros samples
        for i in range(n_samples - 1):
            model.sample(scale=10)
            predictions = np.concatenate(
                (
                    predictions,
                    predict(
                        loader=loader,
                        model=model,
                        loader_scaler=self.loader_scaler,
                    )["predictions"][..., None],
                ),
                axis=2,
            )

        stds = np.std(predictions, axis=2)
        means = np.mean(predictions, axis=2)

        return Y, means, stds

    def _laplace(self, loader, n_samples=25):
        la = self.models[1]

        la.sample(scale=1.0, cov=False)
        init = predict(
            loader=loader, model=la.net, loader_scaler=self.loader_scaler
        )
        predictions = init["predictions"][..., None]
        Y = init["targets"]

        # Outros samples
        for i in range(n_samples - 1):
            la.sample(scale=1.0, cov=False)
            predictions = np.concatenate(
                (
                    predictions,
                    predict(loader=loader, model=la.net)["predictions"][
                        ..., None
                    ],
                ),
                axis=2,
            )

        stds = np.std(predictions, axis=2)
        means = np.mean(predictions, axis=2)

        return Y, means, stds

    def _swag(self, loader, n_samples=25):
        return self._swag_sampler_method(
            self.models[1],
            loader,
            n_samples=n_samples,
        )

    def _multiswag(self, loader, n_samples=20):
        Ys = []
        Ms = []
        STs = []
        for model in self.models[1:]:
            Y, means, stds = self._swag_sampler_method(
                model, loader, n_samples=n_samples
            )
            Ys.append(Y[..., None])
            Ms.append(means[..., None])
            STs.append(stds[..., None])
        Y = np.concatenate(Ys, axis=2).mean(axis=2)
        means = np.concatenate(Ms, axis=2).mean(axis=2)
        stds = np.concatenate(STs, axis=2).mean(axis=2)

        return Y, means, stds

    def _dropout(self, loader, n_samples=25):
        init = predict(
            loader=loader,
            model=self.models[0],
            use_training_true=True,
            loader_scaler=self.loader_scaler,
        )
        predictions = init["predictions"][..., None]
        Y = init["targets"]
        # Outros samples
        for i in range(n_samples - 1):
            predictions = np.concatenate(
                (
                    predictions,
                    predict(
                        loader=loader,
                        model=self.models[0],
                        use_training_true=True,
                        loader_scaler=self.loader_scaler,
                    )["predictions"][..., None],
                ),
                axis=2,
            )

        stds = np.std(predictions, axis=2)
        means = np.mean(predictions, axis=2)

        return Y, means, stds

    def _ensemble(self, loader):
        means_ = []
        variances_ = []
        for model in self.models:
            preds = predict_gaussian(
                loader=loader, model=model, loader_scaler=self.loader_scaler
            )
            means, variances = preds["predictions"]
            means_.append(means[..., None])
            variances_.append(variances[..., None])

        means_f = np.concatenate(means_, axis=2).mean(axis=2)
        variances_f = (
            (np.concatenate(means_, axis=2) ** 2).mean(axis=2)
            + np.concatenate(variances_, axis=2).mean(axis=2)
            - means_f**2
        )
        Y = preds["targets"]

        return Y, means_f, np.sqrt(variances_f)

    def _dummy(self, loader, n_samples=3):
        init = predict(
            loader=loader,
            model=self.models[0],
            loader_scaler=self.loader_scaler,
        )
        predictions = init["predictions"][..., None]
        Y = init["targets"]

        # Outros samples
        for i in range(n_samples - 1):
            predictions = np.concatenate(
                (
                    predictions,
                    predict(
                        loader=loader,
                        model=self.models[0],
                        loader_scaler=self.loader_scaler,
                    )["predictions"][..., None],
                ),
                axis=2,
            )

        stds = np.std(predictions, axis=2)
        means = np.mean(predictions, axis=2)

        return Y, means, stds

    def _nllbaseline(self, loader):
        model = self.models[0]
        preds = predict_gaussian(
            loader=loader, model=model, loader_scaler=self.loader_scaler
        )
        means, variances = preds["predictions"]
        Y = preds["targets"]
        return Y, means, np.sqrt(variances)

    def sample(self, loader, n_samples=25):
        if self.model_type == "swag":
            return self._swag(loader, n_samples=n_samples)
        elif self.model_type == "laplace":
            return self._laplace(loader, n_samples=n_samples)
        elif self.model_type == "multiswag":
            return self._multiswag(loader, n_samples=n_samples)
        elif self.model_type == "dropout":
            return self._dropout(loader, n_samples=n_samples)
        elif self.model_type == "ensemble":
            return self._ensemble(loader)
        elif self.model_type == "dummy":
            return self._dummy(loader)
        elif self.model_type == "nllbaseline":
            return self._nllbaseline(loader)
        else:
            raise Exception(
                f"O model_type {self.model_type} não tem Sampler associado"
            )

    def _ensemble_input(self, input):
        means_ = []
        variances_ = []
        with torch.no_grad():
            for model in self.models:
                means, variances = model(input)
                means_.append(means[..., None])
                variances_.append(variances[..., None])

        means_f = np.concatenate(means_, axis=2).mean(axis=2)
        variances_f = (
            np.concatenate(means_, axis=2).mean(axis=2) ** 2
            + np.concatenate(variances_, axis=2).mean(axis=2)
            - means_f**2
        )

        return means_f, np.sqrt(variances_f)

    def _ensemble_input_sample(self, input):
        means, stds = self._ensemble_input(input[None, ...])
        return (means + (torch.randn(list(means.shape)) * stds).numpy())[
            0, :, :
        ]

    def simple_sample(self, Y, means, stds, loader=None):
        if loader is not None:
            with torch.no_grad():
                Y, means, stds = self.sample(loader, n_samples=25)
        else:
            pass
        sample = means + (torch.randn(list(means.shape)) * stds).numpy()
        return Y, means, sample

    def simple_sample_with_dataset(self, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        with torch.no_grad():
            Y, means, stds = self.sample(loader, n_samples=25)
            sample = means + (torch.randn(list(means.shape)) * stds).numpy()
        return Y, means, sample


# %% Train NLL Model

model = GaussianBase(base_class=ToyModel)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,  # , momentum=args.momentum, weight_decay=args.wd
)


def gaussian_nll(model, input, target):
    lossfn = F.gaussian_nll_loss
    output, variance = model(input)
    loss = lossfn(output, target, variance)

    return loss, output


criterion = gaussian_nll  # Precisa ser treinada com saída NLL

train_loss_array, test_loss_array = [], []

for epoch in range(epochs):
    train_res = train_epoch(
        loaders["train"],
        model,
        criterion,
        optimizer,
    )
    test_res = eval(
        loaders["test"],
        model,
        criterion,
    )

    values = [
        epoch + 1,
        learning_rate,
        train_res["loss"],
        test_res["loss"],
        None,
        1.0,
    ]

    test_loss_array.append(test_res["loss"])
    train_loss_array.append(train_res["loss"])

    print_tabulate(values, epoch)


# %%
sampler = BayesianModelPredictor(
    models=[model],
    model_type="nllbaseline",
)

Y, means, stds = sampler.sample(loaders["test"])

# %%
plot_loss(train_loss_array, test_loss_array)
# %%
plot(dataset_train, dataset_test, Y, means, stds, label="NLL Baseline")


# %% Train Deep Ensembles

models = [GaussianBase(base_class=ToyModel) for i in range(0, 5)]

for model in models:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,  # , momentum=args.momentum, weight_decay=args.wd
    )
    criterion = gaussian_nll  # Precisa ser treinada com saída NLL

    train_loss_array, test_loss_array = [], []

    for epoch in range(epochs):
        train_res = train_epoch(
            loaders["train"],
            model,
            criterion,
            optimizer,
        )
        test_res = eval(
            loaders["test"],
            model,
            criterion,
        )

        values = [
            epoch + 1,
            learning_rate,
            train_res["loss"],
            test_res["loss"],
            None,
            1.0,
        ]

        test_loss_array.append(test_res["loss"])
        train_loss_array.append(train_res["loss"])

        print_tabulate(values, epoch)

# %%
sampler = BayesianModelPredictor(
    models=models,
    model_type="ensemble",
)

Y, means, stds = sampler.sample(loaders["test"])

# %%
plot_loss(train_loss_array, test_loss_array)
# %%
plot(dataset_train, dataset_test, Y, means, stds, label="Ensemble")

# %%
# %% Train Dropout Model

model = ToyModel(dropout=0.3)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,  # , momentum=args.momentum, weight_decay=args.wd
)


def mse_loss(model, input, target):
    output = model(input)
    loss = F.mse_loss(output, target)
    return loss, output


criterion = mse_loss  # Precisa ser treinada com saída NLL

train_loss_array, test_loss_array = [], []

for epoch in range(epochs):
    train_res = train_epoch(
        loaders["train"],
        model,
        criterion,
        optimizer,
    )
    test_res = eval(
        loaders["test"],
        model,
        criterion,
    )

    values = [
        epoch + 1,
        learning_rate,
        train_res["loss"],
        test_res["loss"],
        None,
        1.0,
    ]

    test_loss_array.append(test_res["loss"])
    train_loss_array.append(train_res["loss"])

    print_tabulate(values, epoch)


# %%
sampler = BayesianModelPredictor(
    models=[model],
    model_type="dropout",
)

Y, means, stds = sampler.sample(loaders["test"])

# %%
plot_loss(train_loss_array, test_loss_array)
# %%
plot(dataset_train, dataset_test, Y, means, stds, label="Dropout")


# %% Train SWAG Model
from subspace_inference.posteriors.swag import SWAG

swag_learning_rate = 0.01

swag_models = [
    SWAG(
        ToyModel,
        subspace_type="pca",
        subspace_kwargs={"max_rank": 10, "pca_rank": 10},
    )
    for _ in range(5)
]

criterion = mse_loss

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=swag_learning_rate,  # , momentum=args.momentum, weight_decay=args.wd
    # weight_decay=args.wd,
)
for swag_model in swag_models:
    for epoch in range(25 * 5):
        train_res = train_epoch(
            loaders["train"],
            model,
            criterion,
            optimizer,
        )
        # test_res = utils.eval(loaders["test"], model, criterion, cuda=False, regression=True)
        test_res = {"loss": None}

        if epoch % 5 == 0:
            swag_model.collect_model(model)

            # Eval Model Criterion
            swag_model.sample(0.0)

            swag_res = eval(
                loaders["test"],
                swag_model,
                criterion,
            )

            values = [
                epoch + 1,
                swag_learning_rate,
                train_res["loss"],
                test_res["loss"],
                swag_res["loss"],
                0.1,
            ]

            print_tabulate(values, epoch)

# %%
sampler = BayesianModelPredictor(
    models=[model, swag_model],
    model_type="swag",
)

Y, means, stds = sampler.sample(loaders["test"])

# %%
plot_loss(train_loss_array, test_loss_array)
# %%
plot(dataset_train, dataset_test, Y, means, stds, label="SWAG")


# %%
sampler = BayesianModelPredictor(
    models=[model] + swag_models,
    model_type="multiswag",
)

Y, means, stds = sampler.sample(loaders["test"])

# %%
plot_loss(train_loss_array, test_loss_array)
# %%
plot(dataset_train, dataset_test, Y, means, stds, label="MultiSWAG")

# %%
