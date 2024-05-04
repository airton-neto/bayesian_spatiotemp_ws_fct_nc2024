# %%
import random

import matplotlib.ticker as ticker
import numpy as np
import scipy.stats as st
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from swag import utils


# %%
def plot_loss(train_loss_array, test_loss_array, save_directory=None):
    plt.plot(train_loss_array[3:], color="red", label="Train")
    plt.plot(test_loss_array[3:], color="blue", label="Test")
    plt.legend()
    if save_directory is not None:
        plt.savefig(save_directory)


# %%
def test_prediction(model, loaders, save_directory=None, loader_scaler=None):
    preds_aux = utils.predict(
        loaders["test"], model, loader_scaler=loader_scaler
    )
    index = random.randint(0, preds_aux["predictions"].shape[0])
    fig, ax = plt.subplots(figsize=[16, 5])
    ax.plot(
        preds_aux["predictions"][index, :, 0].cpu(), color="blue", label="pred"
    )
    ax.plot(
        preds_aux["targets"][index, :, 0].cpu(), color="red", label="target"
    )
    ax.legend()

    if save_directory is not None:
        fig.savefig(save_directory)

    preds_aux = utils.predict(
        loaders["train"], model, loader_scaler=loader_scaler
    )
    index = random.randint(0, preds_aux["predictions"].shape[0])
    fig, ax = plt.subplots(figsize=[16, 5])
    ax.plot(
        preds_aux["predictions"][index, :, 0].cpu(), color="blue", label="pred"
    )
    ax.plot(
        preds_aux["targets"][index, :, 0].cpu(), color="red", label="target"
    )
    ax.legend()

    if save_directory is not None:
        fig.savefig(save_directory.split(".png")[0] + "_train.png")


# %% Talagrad cumulative
def plot_talagrad_cumulative(Y, means, stds, save_directory=None):
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
        means = means.cpu().numpy()
        stds = stds.cpu().numpy()
    quantiles = list(np.arange(0.05, 1, 0.05))
    ideal = quantiles
    main_quantiles = list(np.arange(0.1, 1, 0.1))

    if Y is None:
        # usadp pra fzr um exemplo
        perturbado = np.random.uniform(-0.032, 0.032, len(quantiles))
        perturbado[0] = 0
        perturbado[-1] = 0
        cumulatives = np.flip(np.array(quantiles + perturbado))

    cumulatives = []
    for quantile in quantiles:
        cumulatives.append((Y < means - st.norm.ppf(quantile) * stds).sum())

    cumulatives = cumulatives / max(cumulatives)

    fig, ax = plt.subplots(figsize=[5, 5])
    ax.plot(quantiles, ideal, linestyle="--", color="#bdbdbd", linewidth=3)
    realizado = np.flip(cumulatives)
    ax.plot(
        quantiles,
        realizado,
        linestyle="--",
        marker="o",
        color="red",
    )
    ax.set_xticks(main_quantiles)
    ax.set_ylabel("Empirical")
    ax.set_xlabel("Nominal")

    if save_directory is not None:
        fig.savefig(save_directory)
        # fig.savefig("../Figuras/talagrad_cumulatve_example.png")


# %% Talagrad Sharpness
def get_ticker_label(x, pos):
    return f"{int(x)}h"


def plot_talagrad_sharpness(stds, save_directory=None, seq_length=24):
    # Gráfico de sharpness com lead time
    # Sharpness = P90 - P10, médio. Isso é feito para cada horário do lead time.
    main_quantiles = [0.1, 0.2, 0.3, 0.4]
    colors = ["#ced4da", "#adb5bd", "#6c757d", "#343a40"]

    fig, ax = plt.subplots(figsize=[5, 5])

    for quantile, color in zip(main_quantiles, colors):
        intervals = (
            st.norm.ppf(1 - quantile) * stds - st.norm.ppf(quantile) * stds
        )
        ax.plot(
            np.arange(0, seq_length, 1) + 1,
            intervals.mean(axis=2).mean(axis=0),
            color=color,
        )
    ax.set_ylim(0, int(intervals.mean(axis=2).mean(axis=0).max() + 4))
    ax.set_xlim(1, seq_length)
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("Sharpness (m/s)")
    ax.xaxis.set_ticks(range(0, seq_length + 1, 8))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(get_ticker_label))

    custom_lines = [Line2D([], [], color=color, lw=2) for color in colors]

    ax.legend(
        custom_lines,
        ["P-60", "P-70", "P-80", "P-90"],
        loc="lower center",
        bbox_to_anchor=(0.48, -0.2),
        # loc=0,
        labelspacing=0.1,
        columnspacing=1.5,
        frameon=False,
        # fontsize=7,
        ncol=4,
    )

    if save_directory is not None:
        fig.savefig(save_directory, bbox_inches="tight")
        # fig.savefig("../Figuras/talagrad_cumulatve_example.png", , bbox_inches='tight')


# %% Talagrad Sharpness
def get_ticker_label(x, pos):
    return f"{int(x)}h"


def plot_talagrad_sharpness_example(seq_length=24):
    # Gráfico de sharpness com lead time
    # Sharpness = P90 - P10, médio. Isso é feito para cada horário do lead time.
    main_quantiles = [0.1, 0.2, 0.3, 0.4]
    colors = ["#ced4da", "#adb5bd", "#6c757d", "#343a40"]
    perturbation = np.random.uniform(-0.1, 0.1, seq_length)

    fig, ax = plt.subplots(figsize=[5, 5])

    for quantile, color in zip(main_quantiles, colors):
        intervals = quantile * np.ones(seq_length) * 5 + perturbation
        ax.plot(
            np.arange(0, seq_length, 1) + 1,
            intervals,
            color=color,
        )
    ax.set_ylim(0, int(max(intervals) + 4))
    ax.set_xlim(1, seq_length)
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("Sharpness (m/s)")
    ax.xaxis.set_ticks(range(0, seq_length + 1, 8))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(get_ticker_label))

    custom_lines = [Line2D([], [], color=color, lw=2) for color in colors]

    ax.legend(
        custom_lines,
        ["IQ-10", "IQ-20", "IQ-30", "IQ-40"],
        loc="lower center",
        bbox_to_anchor=(0.48, -0.2),
        # loc=0,
        labelspacing=0.1,
        columnspacing=1.5,
        frameon=False,
        # fontsize=7,
        ncol=4,
    )

    fig.savefig(
        "../Figuras/talagrad_sharpness_example.png", bbox_inches="tight"
    )


# %% Plota um exemplo de timeseries
def get_ticker_label(x, pos):
    # return f"D+{x//8}"
    return f"{x}h"


def plot_timeseries_usage_example(
    sampler,
    loader,
    Y,
    means,
    stds,
    save_directory=None,
    dois_desvios=False,
    seq_length=24,
):
    z = 1.28 if not dois_desvios else 2  # IC de 80% ou dois desvios

    if isinstance(Y, torch.Tensor):
        Y = Y.cpu()
        means = means.cpu()
        stds = stds.cpu()

    # Pegando o vento total
    Y_ = np.sqrt(Y[:, :, 0] ** 2 + Y[:, :, 1] ** 2)
    means_ = np.sqrt(means[:, :, 0] ** 2 + means[:, :, 1] ** 2)
    stds_ = np.sqrt(stds[:, :, 0] ** 2 + stds[:, :, 1] ** 2)

    # Define um index qualquer para buscar a hora que vai buscar os dados
    general_index = 0

    samples = []
    samples_ = []
    for i in range(5):
        _, _, sample = sampler.simple_sample(Y_, means_, stds_)
        # samples.append(sample.copy())
        # sample_ = np.sqrt(sample[:, :, 0] ** 2 + sample[:, :, 1] ** 2)
        samples_.append(sample)

    fig, ax = plt.subplots(figsize=[16, 5])
    for sample_ in samples_:
        ax.plot(
            sample_[general_index, :],
            label="Sample",
            color="#828385",
            alpha=0.8,
        )
    ax.plot(Y_[general_index, :], label="Measured", color="red")
    ax.plot(means_[general_index, :], label="Predicted", color="black")
    ax.fill_between(
        list(range(seq_length)),
        means_[general_index, :] - z * stds_[general_index, :],
        means_[general_index, :] + z * stds_[general_index, :],
        color="#d1d1d1",
    )
    ax.set_ylim([0, 12])
    ax.set_xlim([0, seq_length])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks(range(0, seq_length + 1, 8))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(get_ticker_label))
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylabel("Wind Speed (m/s)", fontsize=17)

    custom_lines = [
        Line2D([], [], color="red", lw=2),
        Line2D([], [], color="black", lw=2),
        Line2D([], [], color="#828385", lw=2),
    ]
    ax.legend(
        custom_lines,
        ["Measured", "Predicted", "Samples"],
        loc="lower center",
        bbox_to_anchor=(0.48, -0.25),
        # loc=0,
        labelspacing=0.1,
        columnspacing=1.5,
        frameon=False,
        fontsize=17,
        ncol=3,
    )

    if save_directory is not None:
        fig.savefig(save_directory, bbox_inches="tight")


# %% Plota o mapinha com as predições do vento isoladas
def plot_map_with_wind_predictions(
    sampler,
    loader,
    dataset_test,
    Y,
    means,
    stds,
    wtg_data,
    save_directory=None,
):
    z = 1.28  # IC de 80%

    if isinstance(Y, torch.Tensor):
        Y = Y.cpu()
        means = means.cpu()
        stds = stds.cpu()

    # Define um index qualquer para buscar a hora que vai buscar os dados
    general_index = 0
    hour_forecast = 5
    scale = 13

    # Busca o horário do index escolhido. Procura os outros indexes das outras turbinas, no mesmo horário.
    first_idx = dataset_test.indexes[general_index]
    hour = wtg_data.loc[first_idx, "hour"]
    indexes = list(dataset_test.data[wtg_data.hour == hour].index)

    # dtsidxs = []
    # devices_ids = []
    # for index in indexes:
    #     try:
    #         dtsidx = dataset_test.indexes.index(index)
    #         device_id = dataset_test.data.loc[dataset_test.indexes[dtsidx], :][
    #             "device_id"
    #         ]
    #         dtsidxs.append(dtsidx)
    #         devices_ids.append(device_id)
    #     except:
    #         print("not found one index")
    #     inp, out = dataset_test[dtsidx]

    positions = {
        1101: [485, 396],
        1102: [477, 460],
        1103: [468, 524],
        1104: [458, 588],
        1105: [447, 653],
        1106: [435, 731],
        1107: [425, 796],
        1108: [418, 861],
        1301: [624, 333],
        1302: [636, 396],
        1303: [636, 458],
        1304: [644, 523],
        1305: [645, 601],
        1306: [646, 665],
        1307: [635, 730],
        1308: [626, 808],
        1309: [617, 873],
        1310: [598, 938],
    }

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    im = plt.imread("../Figuras/turbine-grid.jpeg")
    fig, ax = plt.subplots(figsize=[5, 5])
    ax.imshow(im, extent=[0, 1000, 0, 1000])

    for index, device_id in enumerate(positions.keys()):
        pos = positions[device_id]
        vxs = []
        vys = []
        for i in range(4):
            Y, means, sample = sampler.simple_sample_with_dataset(dataset_test)
            vx, vy = sample[index, hour_forecast, :]
            ax.arrow(
                *pos, -vx * scale, -vy * scale, color="#999999", alpha=0.7
            )
            vxs.append(vx)
            vys.append(vy)
        ax.arrow(
            *pos, *(-means[index, hour_forecast, :] * scale), color="black"
        )
        ax.arrow(*pos, *(Y[index, hour_forecast, :] * -scale), color="red")
        ax.set_ylim([250, 1000])
        ax.set_xlim([250, 900])
        ax.set_xticks([])
        ax.set_yticks([])
    for frame in ["top", "bottom", "left", "right"]:
        ax.spines[frame].set_visible(False)

    custom_lines = [
        Line2D([], [], color="red", lw=2),
        Line2D([], [], color="black", lw=2),
        Line2D([], [], color="#828385", lw=2),
    ]
    ax.legend(
        custom_lines,
        ["Measured", "Predicted", "Samples"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        # loc=0,
        labelspacing=2,
        columnspacing=1.5,
        frameon=False,
        fontsize=8,
        ncol=3,
    )

    if save_directory is not None:
        fig.savefig(save_directory, bbox_inches="tight")


# %%
