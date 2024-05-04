# %%
import os

import pandas as pd
import torch

from forecast.model import BayesianModelPredictor, get_metrics
from forecast.plot import (
    plot_loss,
    plot_map_with_wind_predictions,
    plot_talagrad_cumulative,
    plot_talagrad_sharpness,
    plot_timeseries_usage_example,
    test_prediction,
)
from forecast.utils import unscale
from swag.posteriors import KFACLaplace


# %%
def load_data(directory):
    """
    Load de um raw dataset de um diretório. Retorna os dados de FCT e de WTG e seus scalers.
    fct_tensor, fct_scaler, wtg_data, wtg_scaler = load_data(directory)
    """
    directory = f"forecast/saved_datasets/{directory}"
    fct_tensor = torch.load(f"{directory}/fct_tensor.pt")
    with open(f"{directory}/fct_dts.json") as f:
        fct_dts = eval(f.read())
    wtg_data = pd.read_pickle(f"{directory}/wtg_data.pkl")

    return fct_tensor, fct_dts, wtg_data


# %%
def save_loaders(
    loaders,
    dataset_train_val,
    dataset_train,
    dataset_val,
    dataset_test,
    loader_info,
):
    directory = f"forecast/saved_loaders/Loader_{loader_info}"
    try:
        os.mkdir(directory)
    except:
        pass
    torch.save(loaders["train"], f"{directory}/train_dataloader.pt")
    torch.save(loaders["val"], f"{directory}/val_dataloader.pt")
    torch.save(loaders["train_val"], f"{directory}/train_val_dataloader.pt")
    torch.save(loaders["test"], f"{directory}/test_dataloader.pt")
    torch.save(dataset_train, f"{directory}/train_dataset.pt")
    torch.save(dataset_val, f"{directory}/val_dataset.pt")
    torch.save(dataset_train_val, f"{directory}/train_val_dataset.pt")
    torch.save(dataset_test, f"{directory}/test_dataset.pt")

    with open(f"{directory}/info.txt", "wb") as f:
        f.write(loader_info.encode())
    print(directory.split("/")[-1])
    return directory


# %%
def load_loaders(directory):
    dataset_train = torch.load(
        f"forecast/saved_loaders/{directory}/train_dataset.pt"
    )
    dataset_val = torch.load(
        f"forecast/saved_loaders/{directory}/val_dataset.pt"
    )
    dataset_train_val = torch.load(
        f"forecast/saved_loaders/{directory}/train_val_dataset.pt"
    )
    dataset_test = torch.load(
        f"forecast/saved_loaders/{directory}/test_dataset.pt"
    )
    loaders = {
        "train": torch.load(
            f"forecast/saved_loaders/{directory}/train_dataloader.pt"
        ),
        "val": torch.load(
            f"forecast/saved_loaders/{directory}/val_dataloader.pt"
        ),
        "train_val": torch.load(
            f"forecast/saved_loaders/{directory}/train_val_dataloader.pt"
        ),
        "test": torch.load(
            f"forecast/saved_loaders/{directory}/test_dataloader.pt"
        ),
    }
    with open(f"forecast/saved_loaders/{directory}/info.txt", "r") as f:
        print(f.read())

    return loaders, dataset_train_val, dataset_train, dataset_val, dataset_test


# %%
def save_model_information(model_info):
    file = "model_summary.xlsx"
    if os.path.exists(file):
        with pd.ExcelWriter(
            file,
            if_sheet_exists="overlay",
            engine="openpyxl",
            mode="a",
        ) as writer:
            pd.DataFrame({k: [v] for k, v in model_info.items()}).to_excel(
                writer, index=False
            )
    else:
        pd.DataFrame({k: [v] for k, v in model_info.items()}).to_excel(
            file, index=False
        )


# %%
def save_model_information(model_info):
    model_name = model_info.get("model")
    file = f"summary_{model_name}.xlsx"

    new_data = pd.DataFrame({k: [v] for k, v in model_info.items()})
    new_data[
        [
            "skip_metrics",
            "skip_plots",
            "grid_search",
        ]
    ] = new_data[
        [
            "skip_metrics",
            "skip_plots",
            "grid_search",
        ]
    ].astype(int)

    if "swag_learning_rate" not in new_data.columns:
        new_data["swag_learning_rate"] = ""

    columns = [
        "model",
        "bayesian_framework",
        "dataset",
        "n_hidden",
        "dropout_rate",
        "swag_learning_rate",
        "rmse",
        "nll",
        "crps",
        "skip_metrics",
        "skip_plots",
        "grid_search",
        "directory",
    ]
    new_data = new_data[columns]

    if not os.path.exists(file):
        new_data.to_excel(file, index=False)

    else:
        old_data = pd.read_excel(file)

        pd.concat([old_data, new_data]).sort_values(
            by=["grid_search", "bayesian_framework", "dataset", "rmse"],
        ).reset_index(drop=True).to_excel(file, index=False)


# %%
def save_model(
    bayesian_framework,
    models,
    loaders,
    model_information,
    loss_arrays=[],
    calc_metrics=False,
    plot_graphs=True,
    dataset_test=None,
    dataset_path="",
    loader_scaler=None,
    loader_path=None,
):
    assert bayesian_framework in (
        "swag",
        "multiswag",
        "dropout",
        "laplace",
        "ensemble",
        "dummy",
        "nllbaseline",
    )
    fct_tensor, fct_dts, wtg_data = load_data(dataset_path)
    loaders_app, _, dataset_app, _, _ = load_loaders("Loader_App")
    model_info = model_information.copy()
    model_info["bayesian_framework"] = bayesian_framework
    directory = (
        f"forecast/saved_models/{model_info['model']}_{bayesian_framework}_"
        + str(pd.Timestamp.now(tz="America/Fortaleza"))
    )
    print(f"Saved in {directory}")
    os.mkdir(directory)
    os.mkdir(f"{directory}/plots")
    model_info["directory"] = directory.split("/")[-1]

    if bayesian_framework == "swag":
        torch.save(models[0], f"{directory}/model.pt")
        torch.save(models[1], f"{directory}/swag.pt")
    if bayesian_framework == "laplace":
        torch.save(models[0], f"{directory}/model.pt")
        torch.save(models[1], f"{directory}/laplace.pt")
    if bayesian_framework == "multiswag":
        torch.save(models[0], f"{directory}/model.pt")
        [
            torch.save(model, f"{directory}/swag{i}.pt")
            for i, model in enumerate(models[1:])
        ]
    elif bayesian_framework == "dropout":
        torch.save(models[0], f"{directory}/dropout.pt")
    elif bayesian_framework == "dummy":
        torch.save(models[0], f"{directory}/dummy.pt")
    elif bayesian_framework == "ensemble":
        [
            torch.save(model, f"{directory}/model{i}.pt")
            for i, model in enumerate(models)
        ]
    elif bayesian_framework == "nllbaseline":
        torch.save(models[0], f"{directory}/model.pt")

    # torch.save(loaders["train"], f"{directory}/train_dataloader.pt")
    # torch.save(loaders["test"], f"{directory}/test_dataloader.pt")

    with open(f"{directory}/info.txt", "wb") as f:
        f.write(str(model_info).encode())

    with open(f"{directory}/loader_path.txt", "wb") as f:
        f.write(str(loader_path).encode())

    if loss_arrays:
        plot_loss(
            *loss_arrays, save_directory=f"{directory}/plots/loss_array.png"
        )

    if bayesian_framework not in ("ensemble", "nllbaseline"):
        test_prediction(
            models[0],
            loaders,
            save_directory=f"{directory}/plots/test_prediction.png",
            loader_scaler=loader_scaler,
        )

    if calc_metrics:
        sampler = BayesianModelPredictor(
            models, bayesian_framework, loader_scaler
        )
        Y, means, stds = sampler.sample(loaders["test"])
        rmse, nll, crps = get_metrics(Y, means, stds)
    else:
        rmse, nll, crps = None, None, None

    model_info.update(
        {
            "rmse": rmse,
            "nll": nll,
            "crps": crps,
            "created_at": pd.Timestamp.now(tz=None),
        }
    )

    save_model_information(model_info)

    # if plot_graphs:
    #     plot_talagrad_cumulative(
    #         Y,
    #         means,
    #         stds,
    #         save_directory=f"{directory}/plots/talagrad_cumulative.png",
    #     )
    #     plot_talagrad_sharpness(
    #         stds,
    #         save_directory=f"{directory}/plots/talagrad_sharpness.png",
    #     )

    #     Y, means, stds = sampler.sample(loaders_app["test"])

    #     plot_timeseries_usage_example(
    #         sampler,
    #         loaders_app["test"],
    #         Y,
    #         means,
    #         stds,
    #         save_directory=f"{directory}/plots/timeseries_usage_example.png",
    #     )
    # plot_timeseries_usage_example(
    #     sampler,
    #     loaders_app["test"],
    #     Y,
    #     means,
    #     stds,
    #     save_directory=f"{directory}/plots/timeseries_usage_example_dois_desvios.png",
    #     dois_desvios=True,
    # )
    # plot_map_with_wind_predictions(
    #     sampler,
    #     loaders_app["test"],
    #     dataset_app,
    #     Y,
    #     means,
    #     stds,
    #     wtg_data,
    #     save_directory=f"{directory}/plots/map_with_predictions.png",
    # )

    return directory


# %%
def load_model(directory):
    directory = directory.split("/")[-1]
    bayesian_framework = directory.split("_")[1]

    if bayesian_framework == "swag":
        model = torch.load(f"forecast/saved_models/{directory}/model.pt")
        model.eval()
        swag = torch.load(f"forecast/saved_models/{directory}/swag.pt")
        swag.eval()
        models = [model, swag]
    if bayesian_framework == "laplace":
        model = torch.load(f"forecast/saved_models/{directory}/model.pt")
        model.eval()
        laplace = torch.load(f"forecast/saved_models/{directory}/laplace.pt")
        laplace.eval()
        models = [model, laplace]
    if bayesian_framework == "multiswag":
        model = torch.load(f"forecast/saved_models/{directory}/model.pt")
        model.eval()
        swags = []
        for i in range(100):
            try:
                model = torch.load(
                    f"forecast/saved_models/{directory}/swag{i}.pt"
                )
                model.eval()
                swags.append(model)
            except:
                break
        models = [model] + swags
    elif bayesian_framework == "dropout":
        model = torch.load(f"forecast/saved_models/{directory}/dropout.pt")
        model.eval()
        models = [model]
    elif bayesian_framework == "dummy":
        model = torch.load(f"forecast/saved_models/{directory}/dummy.pt")
        model.eval()
        models = [model]
    elif bayesian_framework == "ensemble":
        models = []
        for i in range(100):
            try:
                model = torch.load(
                    f"forecast/saved_models/{directory}/model{i}.pt"
                )
                model.eval()
                models.append(model)
            except:
                break
    if bayesian_framework == "nllbaseline":
        model = torch.load(f"forecast/saved_models/{directory}/model.pt")
        model.eval()
        models = [model]

    with open(f"forecast/saved_models/{directory}/loader_path.txt", "r") as f:
        loader_path = f.read()

    (
        loaders,
        dataset_train_val,
        dataset_train,
        dataset_val,
        dataset_test,
    ) = load_loaders(loader_path)

    with open(f"forecast/saved_models/{directory}/info.txt", "r") as f:
        print(f.read())

    return models, loaders
