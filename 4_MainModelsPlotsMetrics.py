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
with open("main_models.yaml", "r") as ymlfile:
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


# %%
results = pd.DataFrame(
    columns=["Dataset", "Model", "Bayesian", "RMSE", "NLL", "CRPS"], data=[]
)


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

    for model_nn, _aux2 in _aux1.items():
        for bayesian_framework, model_path in _aux2.items():
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
            rmse, nll, crps = get_metrics(Y, means, stds)

            metrics[dataset_type][model_nn][bayesian_framework] = {
                "rmse": rmse,
                "nll": nll,
                "crps": crps,
            }

            results = results.append(
                pd.DataFrame(
                    [
                        [
                            dataset_type,
                            model_nn,
                            bayesian_framework,
                            rmse,
                            nll,
                            crps,
                        ]
                    ],
                    columns=results.columns,
                ),
                ignore_index=True,
            )

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

            # Y, means, stds = sampler.sample(loaders_app["test"])

            # plot_timeseries_usage_example(
            #     sampler,
            #     loaders_app["test"],
            #     Y,
            #     means,
            #     stds,
            #     save_directory=f"{image_directory}/{prefix}_timeseries_usage_example.png",
            # )
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


# %% Apenas pra debug
results = pd.DataFrame(
    columns=["Dataset", "Model", "Bayesian", "RMSE", "NLL", "CRPS"], data=[]
)

for dataset_type, _aux1 in model_dict.items():
    for model_nn, _aux2 in _aux1.items():
        for bayesian_framework, model_path in _aux2.items():
            results = results.append(
                pd.DataFrame(
                    [
                        [
                            dataset_type,
                            model_nn,
                            bayesian_framework,
                            metrics[dataset_type][model_nn][
                                bayesian_framework
                            ]["rmse"],
                            metrics[dataset_type][model_nn][
                                bayesian_framework
                            ]["nll"],
                            metrics[dataset_type][model_nn][
                                bayesian_framework
                            ]["crps"],
                        ]
                    ],
                    columns=results.columns,
                ),
                ignore_index=True,
            )

# %% Para Cada Dataset
for dataset_type in ["A", "C"]:
    aux = (
        results[results["Dataset"] == dataset_type]
        .pivot_table(
            index=["Model", "Bayesian"],
            values=["RMSE", "NLL", "CRPS"],
            columns=[],
        )
        .reset_index()
    )
    aux["shift"] = aux["Model"].shift(1) == aux["Model"]
    aux["Model"] = aux.apply(
        lambda row: row["Model"] if not row["shift"] else "", axis=1
    )
    aux.drop(columns=["shift"], inplace=True)

    aux2 = aux[aux["Bayesian"] == "dummy"][:1]
    aux = aux[aux["Bayesian"] != "dummy"]
    aux2["Model"] = "Dummy"
    aux2["Bayesian"] = "---"
    aux2["NLL"] = np.nan
    aux2["CRPS"] = np.nan
    aux = aux.append(aux2).reset_index(drop=True)
    aux.columns = ["textbf{" + col + "}" for col in aux.columns]

    message = aux.to_latex(float_format="%.4f", index=False).replace(
        "NaN", "---"
    )
    message = (
        message.replace("Bayesian", "Bayesian Framework")
        .replace("dropout", "MC Dropout")
        .replace("ensemble", "Deep Ensembles")
        .replace("multiswag", "MultiSWAG")
        .replace("swag", "SWAG")
        .replace("nllbaseline", "NLL Baseline")
        .replace("mlp", "MLP")
        .replace("convlstm", "ConvLSTM")
        .replace("lstm", "LSTM")
    )

    print(message.replace("textbf", "\\textbf"))

# %%
