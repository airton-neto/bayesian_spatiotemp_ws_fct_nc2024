# %%
from copy import deepcopy

import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.stats as st
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from yaml.loader import SafeLoader

from forecast.dataset import LoaderScaler
from forecast.get import load_data, load_loaders, load_model
from forecast.model import BayesianModelPredictor

# %% read ymal file and create metrics json
with open("main_models.yaml", "r") as ymlfile:
    model_dict = yaml.load(ymlfile, Loader=SafeLoader)
metrics = deepcopy(model_dict)


# %% Calc all metrics

for dataset_type, _aux1 in model_dict.items():
    # Aqui aproveita pra carregar o dataset e os loaders só uma ves
    fct_tensor, fct_dts, wtg_data = load_data("Dataset")
    loader_path = f"Loader_Dataset{dataset_type}"
    (
        _loaders,
        _dataset_train_val,
        _,
        _,
        _dataset_test,
    ) = load_loaders(loader_path)
    loaders_app, _, dataset_app, _, _ = load_loaders("Loader_App")

    loaders = {
        "train": _loaders["train_val"],
        "test": _loaders["test"],
    }
    dataset_train = _dataset_train_val
    dataset_test = _dataset_test
    loader_scaler = LoaderScaler(
        dataset_train
    )  # scaler usa o dataset de treino

    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(10, 6))
    fig2, axs2 = plt.subplots(nrows=3, ncols=5, figsize=(10, 6))
    fig3, axs3 = plt.subplots(nrows=3, ncols=5, figsize=(10, 6))

    for model_index, (model_nn, _aux2) in enumerate(list(_aux1.items())):
        if model_nn == "dummy":
            continue
        for bayesian_index, (bayesian_framework, model_path) in enumerate(
            list(_aux2.items())
        ):
            axs[model_index, bayesian_index].set_yticks([])
            axs[model_index, bayesian_index].set_xticks([])
            axs2[model_index, bayesian_index].set_yticks([])
            axs2[model_index, bayesian_index].set_xticks([])
            axs3[model_index, bayesian_index].set_yticks([])
            axs3[model_index, bayesian_index].set_xticks([])

    for model_index, (model_nn, _aux2) in enumerate(list(_aux1.items())):
        if model_nn == "dummy":
            continue
        for bayesian_index, (bayesian_framework, model_path) in enumerate(
            list(_aux2.items())
        ):
            print(
                f"Dataset {dataset_type} Modelo {model_nn} Bayesian {bayesian_framework} Path {model_path}"
            )

            # dataset_type = "C"
            # model_nn = "lstm"
            # bayesian_framework = "dropout"
            # model_path = model_dict[dataset_type][model_nn][bayesian_framework]

            if not model_path:
                print("Skipping")
                continue

            # load models
            models, loaders = load_model(model_path)

            # Metrics
            sampler = BayesianModelPredictor(
                models, bayesian_framework, loader_scaler
            )
            Y, means, stds = sampler.sample(loaders["test"])
            Y, means, stds = (
                Y.cpu().numpy(),
                means.cpu().numpy(),
                stds.cpu().numpy(),
            )

            image_directory = "../Figuras/results"

            # CUMULATIVE
            quantiles = list(np.arange(0.05, 1, 0.05))
            ideal = quantiles
            main_quantiles = list(np.arange(0.1, 1, 0.1))
            cumulatives = []
            for quantile in quantiles:
                cumulatives.append(
                    (Y < means - st.norm.ppf(quantile) * stds).sum()
                )
            cumulatives = cumulatives / max(cumulatives)

            axs[model_index, bayesian_index].plot(
                quantiles,
                ideal,
                linestyle="--",
                color="#bdbdbd",
                linewidth=3,
            )
            realizado = np.flip(cumulatives)
            axs[model_index, bayesian_index].plot(
                quantiles,
                realizado,
                linestyle="--",
                # marker="o",
                color="red",
                # markersize=3,
                linewidth=3,
            )
            if bayesian_index == 0:
                axs[model_index, bayesian_index].set_yticks([0, 1])
            if model_index == 2:
                axs[model_index, bayesian_index].set_xticks([0, 1])

            # SHARPNESS
            def get_ticker_label(x, pos):
                return f"{int(x)}h"

            seq_length = 24

            main_quantiles = [0.1, 0.2, 0.3, 0.4]
            colors = ["#ced4da", "#adb5bd", "#6c757d", "#343a40"]
            for quantile, color in zip(main_quantiles, colors):
                intervals = (
                    st.norm.ppf(1 - quantile) * stds
                    - st.norm.ppf(quantile) * stds
                )
                axs2[model_index, bayesian_index].plot(
                    np.arange(0, seq_length, 1) + 1,
                    intervals.mean(axis=2).mean(axis=0),
                    color=color,
                )

            axs2[model_index, bayesian_index].set_ylim(0, 10)
            axs2[model_index, bayesian_index].set_xlim(1, seq_length)
            # axs2[model_index, bayesian_index].set_xlabel("Lead Time")
            # axs2[model_index, bayesian_index].set_ylabel("Sharpness (m/s)")

            custom_lines = [
                Line2D([], [], color=color, lw=2) for color in colors
            ]

            if bayesian_index == 0:
                axs2[model_index, bayesian_index].set_yticks([0, 5])
            if model_index == 2:
                axs2[model_index, bayesian_index].xaxis.set_ticks([0, 24])
                axs2[model_index, bayesian_index].xaxis.set_major_formatter(
                    ticker.FuncFormatter(get_ticker_label)
                )

            # axs2[model_index, bayesian_index].legend(
            #     custom_lines,
            #     ["P-60", "P-70", "P-80", "P-90"],
            #     loc="lower center",
            #     bbox_to_anchor=(0.48, -0.2),
            #     # loc=0,
            #     labelspacing=0.1,
            #     columnspacing=1.5,
            #     frameon=False,
            #     # fontsize=7,
            #     ncol=4,
            # )

            # ERRORS

            real = Y.flatten()
            predito = means.flatten()

            axs3[model_index, bayesian_index].scatter(
                real, predito, alpha=0.07, s=1, color="red"
            )
            axs3[model_index, bayesian_index].plot(
                list(range(-14, 16, 2)),
                list(range(-14, 16, 2)),
                linestyle="--",
                color="#bdbdbd",
                linewidth=2,
            )

            axs3[model_index, bayesian_index].set_xlim([-14, 14])
            axs3[model_index, bayesian_index].set_ylim([-14, 14])

            # Define the bin boundaries using numpy.histogram_bin_edges
            bin_edges = np.histogram_bin_edges(real, bins=range(-14, 16, 2))

            # Calculate the mean and standard deviation for each bin
            bin_means, bin_stds = [], []
            for i in range(len(bin_edges) - 1):
                mask = (real >= bin_edges[i]) & (real < bin_edges[i + 1])
                bin_mean = np.mean(predito[mask])
                bin_std = np.std(predito[mask])
                bin_means.append(bin_mean)
                bin_stds.append(bin_std)

            # Plot the interval plot
            axs3[model_index, bayesian_index].errorbar(
                (bin_edges + (bin_edges[1] - bin_edges[0]) / 2)[0:-1],
                bin_means,
                yerr=np.array(bin_stds) * 1.96,
                fmt="none",
                capsize=2.6,
                capthick=0.8,
                color="blue",
                linewidth=0.8,
            )

            if bayesian_index == 0:
                axs3[model_index, bayesian_index].set_yticks(
                    [-10, -5, 0, 5, 10]
                )
            if model_index == 2:
                axs3[model_index, bayesian_index].set_xticks(
                    [-10, -5, 0, 5, 10]
                )

    # Set Axis names Cumulative
    axs[0, 0].set_ylabel("MLP\n\nSample Rate")
    axs[1, 0].set_ylabel("LSTM\n\nSample Rate")
    axs[2, 0].set_ylabel("ConvLSTM\n\nSample Rate")
    axs[2, 0].set_xlabel("Quantile\n\nMC Dropout")
    axs[2, 1].set_xlabel("Quantile\n\nSWAG")
    axs[2, 2].set_xlabel("Quantile\n\nMultiSWAG")
    axs[2, 3].set_xlabel("Quantile\n\nDeep Ensembles")
    axs[2, 4].set_xlabel("Quantile\n\nNLL Baseline")

    # Set Axis names Sharpness
    axs2[0, 0].set_ylabel("MLP\n\nIQ. (m/s)")
    axs2[1, 0].set_ylabel("LSTM\n\nIQ. (m/s)")
    axs2[2, 0].set_ylabel("ConvLSTM\n\nIQ. (m/s)")
    axs2[2, 0].set_xlabel("Lead Time (h)\n\nMC Dropout")
    axs2[2, 1].set_xlabel("Lead Time (h)\n\nSWAG")
    axs2[2, 2].set_xlabel("Lead Time (h)\n\nMultiSWAG")
    axs2[2, 3].set_xlabel("Lead Time (h)\n\nDeep Ensembles")
    axs2[2, 4].set_xlabel("Lead Time (h)\n\nNLL Baseline")

    # Set Axis names Errors
    axs3[0, 0].set_ylabel("MLP\n\nObserved (m/s)")
    axs3[1, 0].set_ylabel("LSTM\n\nObserved (m/s)")
    axs3[2, 0].set_ylabel("ConvLSTM\n\nObserved (m/s)")
    axs3[2, 0].set_xlabel("Predicted (m/s)\n\nMC Dropout")
    axs3[2, 1].set_xlabel("Predicted (m/s)\n\nSWAG")
    axs3[2, 2].set_xlabel("Predicted (m/s)\n\nMultiSWAG")
    axs3[2, 3].set_xlabel("Predicted (m/s)\n\nDeep Ensembles")
    axs3[2, 4].set_xlabel("Predicted (m/s)\n\nNLL Baseline")

    fig.savefig(
        f"{image_directory}/{dataset_type}_cumulative.png", bbox_inches="tight"
    )
    fig2.savefig(
        f"{image_directory}/{dataset_type}_sharpness.png", bbox_inches="tight"
    )
    fig3.savefig(
        f"{image_directory}/{dataset_type}_errorplot.png", bbox_inches="tight"
    )

# %%
