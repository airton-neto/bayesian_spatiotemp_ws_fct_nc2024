import numpy as np
import tabulate
from scipy.stats import rankdata
from torch.nn import functional as F

from swag.utils import adjust_learning_rate


def test_dataset_for_model(model, dataset):
    # Testa de o modelo tem um output coerente
    a = model.forward(dataset[0][0][None, :, :])

    b = len(dataset)

    for i, (input, target) in enumerate(dataset):
        print("Input", i, input.shape, "Output", target.shape)
        break

    # Testa se o modelo calcula Loss corretamente
    for i, (input, target) in enumerate(dataset):
        output = model(input)
        loss = F.mse_loss(output, target)


def print_tabulate(values, epoch, scale=1):
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


def learning_schedule_lstm(optimizer, epoch):
    lr = 1e-4
    # if epoch > 180:
    #     lr = 0.0005
    # if epoch > 220:
    #     lr = 0.0001
    adjust_learning_rate(optimizer, lr)
    return optimizer, lr


def learning_schedule_3(optimizer, epoch):
    lr = 0.001
    if epoch > 120:
        lr = 0.0005
    if epoch > 160:
        lr = 0.0001
    adjust_learning_rate(optimizer, lr)
    return optimizer, lr


def learning_schedule_gaussian_mlp(optimizer, epoch):
    lr = 0.001
    if epoch > 40:
        lr = 0.0001
    if epoch > 100:
        lr = 0.00001
    adjust_learning_rate(optimizer, lr)
    return optimizer, lr


def unscale(Y, means, stds, wtg_scaler):
    # De-scaling
    Y[:, :, 0] = (
        Y[:, :, 0] * wtg_scaler["ws_x"]["max"] + wtg_scaler["ws_x"]["mean"]
    )
    Y[:, :, 1] = (
        Y[:, :, 1] * wtg_scaler["ws_y"]["max"] + wtg_scaler["ws_y"]["mean"]
    )
    means[:, :, 0] = (
        means[:, :, 0] * wtg_scaler["ws_x"]["max"] + wtg_scaler["ws_x"]["mean"]
    )
    means[:, :, 1] = (
        means[:, :, 1] * wtg_scaler["ws_y"]["max"] + wtg_scaler["ws_y"]["mean"]
    )
    stds[:, :, 0] = (
        stds[:, :, 0] * wtg_scaler["ws_x"]["max"]
    )  # + wtg_scaler['ws_x']['mean']
    stds[:, :, 1] = (
        stds[:, :, 1] * wtg_scaler["ws_y"]["max"]
    )  # + wtg_scaler['ws_y']['mean']
    return Y, means, stds


def unscale_sample(vec, wtg_scaler):
    vec[:, :, 0] = (
        vec[:, :, 0] * wtg_scaler["ws_x"]["max"] + wtg_scaler["ws_x"]["mean"]
    )
    vec[:, :, 1] = (
        vec[:, :, 1] * wtg_scaler["ws_y"]["max"] + wtg_scaler["ws_y"]["mean"]
    )
    return vec


def rankz(obs, ensemble, mask):
    """Parameters
    ----------
    obs : array of observations
    ensemble : array of ensemble, with the first dimension being the
        ensemble member and the remaining dimensions being identical to obs
    mask : boolean mask of shape of obs, with zero/false being where grid cells are masked.
    Returns
    -------
    histogram data for ensemble.shape[0] + 1 bins.
    The first dimension of this array is the height of
    each histogram bar, the second dimension is the histogram bins.
    """

    mask = np.bool_(mask)

    obs = obs[mask]
    ensemble = ensemble[:, mask]

    combined = np.vstack((obs[np.newaxis], ensemble))

    # print('computing ranks')
    ranks = np.apply_along_axis(
        lambda x: rankdata(x, method="min"), 0, combined
    )

    # print('computing ties')
    ties = np.sum(ranks[0] == ranks[1:], axis=0)
    ranks = ranks[0]
    tie = np.unique(ties)

    for i in range(1, len(tie)):
        index = ranks[ties == tie[i]]
        # print('randomizing tied ranks for ' + str(len(index)) + ' instances where there is ' + str(tie[i]) + ' tie/s. ' + str(len(tie)-i-1) + ' more to go')
        ranks[ties == tie[i]] = [
            np.random.randint(index[j], index[j] + tie[i] + 1, tie[i])[0]
            for j in range(len(index))
        ]

    return np.histogram(
        ranks,
        bins=np.linspace(0.5, combined.shape[0] + 0.5, combined.shape[0] + 1),
    )
