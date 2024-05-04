# %%
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader

from forecast.dataset import LoaderScaler
from forecast.get import load_data, load_loaders, load_model
from forecast.model import BayesianModelPredictor, get_metrics
from forecast.plot import (
    plot_map_with_wind_predictions,
    plot_talagrad_cumulative,
    plot_talagrad_sharpness,
    plot_timeseries_usage_example,
)

# %% read ymal file and create metrics json
with open("wtg_only_models.yaml", "r") as ymlfile:
    model_dict = yaml.load(ymlfile, Loader=SafeLoader)
metrics = deepcopy(model_dict)
for dataset_type, _aux1 in model_dict.items():
    for model_nn, _aux2 in _aux1.items():
        for bayesian_framework, model_path in _aux2.items():
            metrics[dataset_type][model_nn][bayesian_framework] = {
                "rmse": 0.0,
                "nll": 0.0,
                "crps": 0.0,
            }


# %% Calc all metrics
dataset_type = "A"
model_nn = "wtg"
bayesian_framework = "ensemble"

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
loader_scaler = LoaderScaler(dataset_train)  # scaler usa o dataset de treino

# load models
models, loaders = load_model(model_path)

# Plots
prefix = f"{dataset_type}_{model_nn}_{bayesian_framework}"
image_directory = "../Figuras/results"
# plot_talagrad_cumulative(
#     Y,
#     means,
#     stds,
#     save_directory=f"{image_directory}/{prefix}_talagrad_cumulative.png",
# )
# plot_talagrad_sharpness(
#     stds,
#     save_directory=f"{image_directory}/{prefix}_talagrad_sharpness.png",
# )


sampler = BayesianModelPredictor(models, bayesian_framework, loader_scaler)
Y, means, stds = sampler.sample(loaders_app["test"])

plot_timeseries_usage_example(
    sampler,
    loaders_app["test"],
    Y,
    means,
    stds,
    save_directory=f"{image_directory}/{prefix}_timeseries_usage_example.png",
)
# plot_map_with_wind_predictions(
#     sampler,
#     loaders_app["test"],
#     dataset_app,
#     Y,
#     means,
#     stds,
#     wtg_data,
#     save_directory=f"{image_directory}/{prefix}_map_with_predictions.png",
# )

# %%
