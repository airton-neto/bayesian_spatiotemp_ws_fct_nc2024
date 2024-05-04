# %%
import random
from copy import deepcopy

import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from yaml.loader import SafeLoader

from forecast.dataset import LoaderScaler
from forecast.get import load_data, load_loaders, load_model
from forecast.model import BayesianModelPredictor
from swag import utils


def get_ticker_label(x, pos):
    # return f"D+{x//8}"
    return f"{x}h"


# %% read ymal file and create metrics json
# model_dict = {
#     "dropout": "lstm_dropout_2023-07-07 15:15:29.024633-03:00",
#     "swag": "lstm_swag_2023-07-05 13:05:50.648492-03:00",
#     "multiswag": "lstm_multiswag_2023-07-05 13:52:11.551893-03:00",
#     "ensemble": "lstm_ensemble_2023-07-05 19:52:36.096345-03:00",
#     "nllbaseline": "lstm_nllbaseline_2023-07-08 02:48:21.436177-03:00",
# }
model_dict = {
    "dropout": "convlstm_dropout_2023-07-18 19:33:53.827270-03:00",
    "swag": "convlstm_swag_2023-07-19 12:37:59.390245-03:00",
    "multiswag": "convlstm_multiswag_2023-07-20 14:31:46.513519-03:00",
    "ensemble": "convlstm_ensemble_2023-07-20 12:06:12.988727-03:00",
    "nllbaseline": "convlstm_nllbaseline_2023-07-19 11:35:45.012358-03:00",
}
metrics = deepcopy(model_dict)
dataset_type = "A"
image_directory = "../Figuras/results"

# %% Calc all metrics

fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(10, 5 * 3), squeeze=True)

loader_path = f"Loader_Dataset{dataset_type}"
(
    _loaders,
    _dataset_train_val,
    _,
    _,
    _dataset_test,
) = load_loaders(loader_path)
loaders_app, _, dataset_app, _, _ = load_loaders("Loader_App")

fct_tensor, fct_dts, wtg_data = load_data("Dataset")
for bayesian_index, (bayesian_framework, model_path) in enumerate(
    model_dict.items()
):
    # Aqui aproveita pra carregar o dataset e os loaders só uma ves

    seq_length = 24

    loaders = {
        "train": _loaders["train_val"],
        "test": _loaders["test"],
    }
    dataset_train = _dataset_train_val
    dataset_test = _dataset_test
    loader_scaler = LoaderScaler(
        dataset_train
    )  # scaler usa o dataset de treino

    print(f"Bayesian {bayesian_framework} Path {model_path}")

    # dataset_type = "C"
    # model_nn = "lstm"
    # bayesian_framework = "dropout"
    # model_path = model_dict[dataset_type][model_nn][bayesian_framework]

    # load models
    models, loaders = load_model(model_path)

    # Metrics
    sampler = BayesianModelPredictor(models, bayesian_framework, loader_scaler)
    Y, means, stds = sampler.sample(loaders_app["test"])

    z = 1.28  # IC de 80% ou dois desvios

    # Pegando o vento total
    Y_ = torch.sqrt(Y[:, :, 0] ** 2 + Y[:, :, 1] ** 2).cpu()
    means_ = torch.sqrt(means[:, :, 0] ** 2 + means[:, :, 1] ** 2).cpu()
    stds_ = torch.sqrt(stds[:, :, 0] ** 2 + stds[:, :, 1] ** 2).cpu()

    # Define um index qualquer para buscar a hora que vai buscar os dados
    general_index = 0

    samples = []
    samples_ = []
    for i in range(5):
        _, _, sample = sampler.simple_sample(Y_, means_, stds_)
        # samples.append(sample.copy())
        # sample_ = np.sqrt(sample[:, :, 0] ** 2 + sample[:, :, 1] ** 2)
        samples_.append(sample)

    for sample_ in samples_:
        axs[bayesian_index].plot(
            sample_[general_index, :],
            label="Sample",
            color="#828385",
            alpha=0.8,
        )
    axs[bayesian_index].plot(
        Y_[general_index, :], label="Measured", color="red"
    )
    axs[bayesian_index].plot(
        means_[general_index, :], label="Predicted", color="black"
    )
    axs[bayesian_index].fill_between(
        list(range(seq_length)),
        means_[general_index, :] - z * stds_[general_index, :],
        means_[general_index, :] + z * stds_[general_index, :],
        color="#d1d1d1",
    )
    axs[bayesian_index].set_ylim([0, 12])
    axs[bayesian_index].set_xlim([0, seq_length])
    axs[bayesian_index].spines["right"].set_visible(False)
    axs[bayesian_index].spines["top"].set_visible(False)
    axs[bayesian_index].xaxis.set_ticks(range(0, seq_length + 1, 8))
    axs[bayesian_index].yaxis.set_ticks([0, 5, 10])
    axs[bayesian_index].xaxis.set_major_formatter(
        ticker.FuncFormatter(get_ticker_label)
    )
    # axs[bayesian_index].tick_params(axis="x")
    # axs[bayesian_index].tick_params(axis="y")

axs[0].set_ylabel("MC Dropout\n\nWind Speed (m/s)")
axs[1].set_ylabel("SWAG\n\nWind Speed (m/s)")
axs[2].set_ylabel("MultiSWAG\n\nWind Speed (m/s)")
axs[3].set_ylabel("Deep Ensembles\n\nWind Speed (m/s)")
axs[4].set_ylabel("NLL Baseline\n\nWind Speed (m/s)")
axs[4].set_xlabel("Lead Time (h)")

custom_lines = [
    Line2D([], [], color="red", lw=2),
    Line2D([], [], color="black", lw=2),
    Line2D([], [], color="#828385", lw=2),
]
axs[-1].legend(
    custom_lines,
    ["Measured", "Predicted", "Samples"],
    loc="lower center",
    bbox_to_anchor=(0.48, -0.45),
    # loc=0,
    labelspacing=0.1,
    columnspacing=1.5,
    frameon=False,
    # fontsize=17,
    ncol=3,
)

fig.savefig(
    f"{image_directory}/{dataset_type}_usage_example.png", bbox_inches="tight"
)

# %%
