# %% Imports

import argparse
import copy
import sys
import time

import numpy as np
import torch

from forecast.dataset import LoaderScaler
from forecast.get import load_loaders, load_model, save_model
from forecast.model import (
    build_convlstm_model,
    build_dummy_model,
    build_lstm_model,
    build_mlp_model,
    build_nwp_model,
    build_wtg_model,
)
from forecast.utils import (
    learning_schedule_3,
    learning_schedule_lstm,
    print_tabulate,
)
from swag import losses, utils

# %% Args
parser = argparse.ArgumentParser(description="Train all models.")
parser.add_argument("--dataset", type=str, default="A")
parser.add_argument("--model", type=str, default="mlp")
parser.add_argument("--dropout", action="store_true")
parser.add_argument("--swag", action="store_true")
parser.add_argument("--multiswag", action="store_true")
parser.add_argument("--ensemble", action="store_true")
parser.add_argument("--nllbaseline", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--skip_plots", action="store_true")
parser.add_argument("--n_swag_models", type=int, default=25)
parser.add_argument(
    "--epochs", type=int, default=1, help="150 pra MLP, 250 pra lstm"
)
parser.add_argument("--swag_lrs", type=str, default="0.02")
parser.add_argument("--n_hidden", type=int, default=4)
parser.add_argument("--dropout_rate", type=float, default=0.1)
parser.add_argument("--wd", type=float, default=1e-3)
parser.add_argument("--grid_search", action="store_true")
parser.add_argument("--pretrained", type=str, default="")

try:
    args = parser.parse_args()
except:
    args = parser.parse_args(args=[])

# %% Load Data
DATASET_TYPE = args.dataset
DATASET_PATH = "Dataset"
LOADERS = {
    "A": "Loader_DatasetA",
    "B": "Loader_DatasetB",
    "C": "Loader_DatasetC",
    "App": "Loader_App",
}
loader_path = LOADERS[DATASET_TYPE]
(
    _loaders,
    _dataset_train_val,
    _dataset_train,
    _dataset_val,
    _dataset_test,
) = load_loaders(LOADERS[DATASET_TYPE])
loaders_app, _, dataset_app, _, _ = load_loaders(LOADERS["App"])

if args.grid_search:
    print(f"\x1b[1;31mGrid Search Strategy@\x1b[0m")
    loaders = {
        "train": _loaders["train"],
        "test": _loaders["val"],
    }
    dataset_train = _dataset_train
    dataset_test = _dataset_val
else:  # Usar todos os dados para treino
    loaders = {
        "train": _loaders["train_val"],
        "test": _loaders["test"],
    }
    dataset_train = _dataset_train_val
    dataset_test = _dataset_test
loader_scaler = LoaderScaler(dataset_train)  # scaler usa o dataset de treino

model_information = args.__dict__
calc_metrics = not args.skip_metrics
plot_graphs = not args.skip_plots

# %% Dummy Model
if args.model == "dummy":
    print(f"\x1b[1;31mDummy\x1b[0m")
    model = build_dummy_model(dataset_train)
    model_path = save_model(
        "dummy",
        [model],
        loaders,
        model_information,
        calc_metrics=calc_metrics,
        dataset_path=DATASET_PATH,
        loader_scaler=loader_scaler,
        loader_path=loader_path,
        plot_graphs=plot_graphs,
    )
    sys.exit(0)
elif args.model == "lstm":
    print(f"\x1b[1;31mEscolhido modelos com LSTM\x1b[0m")
    build_model_function = build_lstm_model
elif args.model == "mlp":
    print(f"\x1b[1;31mEscolhido modelos com MLP\x1b[0m")
    build_model_function = build_mlp_model
elif args.model == "convlstm":
    print(f"\x1b[1;31mEscolhido modelos com ConvLSTM\x1b[0m")
    build_model_function = build_convlstm_model
elif args.model == "wtg":
    print(f"\x1b[1;31mEscolhido modelo LSTM WTG Only\x1b[0m")
    build_model_function = build_wtg_model
elif args.model == "nwp":
    print(f"\x1b[1;31mEscolhido modelo ConvLSTM NWP Only\x1b[0m")
    build_model_function = build_nwp_model
else:
    raise NotImplementedError("{args.model} is not valid model.")

if args.dataset == "A":
    learning_schedule = learning_schedule_3
else:
    learning_schedule = learning_schedule_lstm


# %% General Dropout Model
if args.pretrained:
    print(f"\x1b[1;31mSelected pre-trained model {args.pretrained}\x1b[0m")
    (model,), _ = load_model(args.pretrained)  # load returns model list
    train_loss_array = [1, 1, 1, 1, 1]
    test_loss_array = [1, 1, 1, 1, 1]

elif args.dropout or args.swag or args.multiswag:
    print(f"\x1b[1;31mTreino do modelo Dropout\x1b[0m")

    model = build_model_function(
        dataset_train, dropout_rate=args.dropout_rate, n_hidden=args.n_hidden
    )

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,  # , momentum=args.momentum,
        weight_decay=args.wd,
    )
    criterion = losses.mse

    epochs = args.epochs

    train_loss_array, test_loss_array = [], []
    trigger_time = 0
    for epoch in range(epochs):
        optimizer, learning_rate = learning_schedule(optimizer, epoch)
        time_ep = time.time()
        train_res = utils.train_epoch(
            loaders["train"],
            model,
            criterion,
            optimizer,
            regression=True,
            verbose=False,
            cuda=True,
            loader_scaler=loader_scaler,
        )
        test_res = utils.eval(
            loaders["test"],
            model,
            criterion,
            cuda=True,
            regression=True,
            loader_scaler=loader_scaler,
        )
        time_ep = time.time() - time_ep

        values = [
            epoch + 1,
            learning_rate,
            train_res["loss"],
            test_res["loss"],
            None,
            time_ep,
        ]

        test_loss_array.append(test_res["loss"])
        train_loss_array.append(train_res["loss"])

        patience = 4
        if epoch >= 5:
            if test_loss_array[-1] > test_loss_array[-2]:
                trigger_time += 1
                if trigger_time >= patience:
                    print("Early stopping")
                    break
            else:
                trigger_time = 0

        print_tabulate(values, epoch)


# %% Save Model
if args.dropout:
    save_model(
        bayesian_framework="dropout",
        models=[model],
        loaders=loaders,
        model_information=model_information,
        loss_arrays=[train_loss_array, test_loss_array],
        calc_metrics=calc_metrics,
        dataset_test=dataset_test,
        dataset_path=DATASET_PATH,
        loader_scaler=loader_scaler,
        loader_path=loader_path,
        plot_graphs=plot_graphs,
    )


# %% SWAG MODEL
if args.swag:
    print(f"\x1b[1;31mTreino do SWAG\x1b[0m")

    for _swag_lr in args.swag_lrs.split(","):
        print(f"\x1b[1;31mTreino do SWAG: Learning Rate {_swag_lr}\x1b[0m")
        _model = copy.deepcopy(model)

        swag_model = build_model_function(
            dataset_train,
            swag=True,
            dropout_rate=args.dropout_rate,
            n_hidden=args.n_hidden,
        )

        criterion = losses.mse

        swag_learning_rate = float(_swag_lr)
        model_information["swag_learning_rate"] = swag_learning_rate
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=swag_learning_rate,  # , momentum=args.momentum, weight_decay=args.wd
            weight_decay=args.wd,
        )

        for epoch in range(args.n_swag_models * 5):
            print(f"Epoch {epoch}")
            time_ep = time.time()
            utils.adjust_learning_rate(optimizer, swag_learning_rate)
            train_res = utils.train_epoch(
                loaders["train"],
                model,
                criterion,
                optimizer,
                regression=True,
                verbose=False,
                cuda=True,
                loader_scaler=loader_scaler,
            )
            # test_res = utils.eval(loaders["test"], model, criterion, cuda=False, regression=True)
            test_res = {"loss": None}
            time_ep = time.time() - time_ep

            if epoch % 5 == 0:
                print("Collecting swag model")
                swag_model.collect_model(model)
                print(f"SWAG models: {swag_model.n_models}")

                # Eval Model Criterion
                swag_model.sample(0.0)
                utils.bn_update(loaders["train"], swag_model)
                swag_res = utils.eval(
                    loaders["test"],
                    swag_model,
                    criterion,
                    cuda=True,
                    regression=True,
                    loader_scaler=loader_scaler,
                )

                values = [
                    epoch + 1,
                    swag_learning_rate,
                    train_res["loss"],
                    test_res["loss"],
                    swag_res["loss"],
                    time_ep,
                ]

                print_tabulate(values, epoch)

        # Save Model
        models = [model, swag_model]

        model_path = save_model(
            bayesian_framework="swag",
            models=models,
            loaders=loaders,
            model_information=model_information,
            loss_arrays=[train_loss_array, test_loss_array],
            calc_metrics=calc_metrics,
            dataset_test=dataset_test,
            dataset_path=DATASET_PATH,
            loader_scaler=loader_scaler,
            loader_path=loader_path,
            plot_graphs=plot_graphs,
        )
        model_path = model_path.split("/")[-1]
        print(model_path)

        model = copy.deepcopy(_model)
        model_information.pop("swag_learning_rate")


# %% MultiSWAG MODEL
if args.multiswag:
    print(f"\x1b[1;31mTreino do MultiSWAG\x1b[0m")

    for _swag_lr in args.swag_lrs.split(","):
        _model = copy.deepcopy(model)

        swag_models = [
            build_model_function(
                dataset_train,
                swag=True,
                dropout_rate=args.dropout_rate,
                n_hidden=args.n_hidden,
            )
            for i in range(5)
        ]

        criterion = losses.mse

        for i, swag_model in enumerate(swag_models):
            print(f"\x1b[1;31mTreino do MultiSWAG - Modelo {str(i)}\x1b[0m")

            swag_learning_rate = float(_swag_lr)
            model_information["swag_learning_rate"] = swag_learning_rate
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=swag_learning_rate,  # , momentum=args.momentum, weight_decay=args.wd
                weight_decay=args.wd,
            )

            for epoch in range(args.n_swag_models * 5):
                time_ep = time.time()
                utils.adjust_learning_rate(optimizer, swag_learning_rate)
                train_res = utils.train_epoch(
                    loaders["train"],
                    model,
                    criterion,
                    optimizer,
                    regression=True,
                    verbose=False,
                    cuda=True,
                    loader_scaler=loader_scaler,
                )
                # test_res = utils.eval(loaders["test"], model, criterion, cuda=False, regression=True)
                test_res = {"loss": None}
                time_ep = time.time() - time_ep

                if epoch % 5 == 0:
                    swag_model.collect_model(model)

                    # Eval Model Criterion
                    swag_model.sample(0.0)
                    utils.bn_update(loaders["train"], swag_model)
                    swag_res = utils.eval(
                        loaders["test"],
                        swag_model,
                        criterion,
                        cuda=True,
                        regression=True,
                        loader_scaler=loader_scaler,
                    )

                    values = [
                        epoch + 1,
                        swag_learning_rate,
                        train_res["loss"],
                        test_res["loss"],
                        swag_res["loss"],
                        time_ep,
                    ]

                    print_tabulate(values, epoch)

        # Save Model
        model_path = save_model(
            bayesian_framework="multiswag",
            models=[model] + swag_models,
            loaders=loaders,
            model_information=model_information,
            loss_arrays=[train_loss_array, test_loss_array],
            calc_metrics=calc_metrics,
            dataset_test=dataset_test,
            dataset_path=DATASET_PATH,
            loader_scaler=loader_scaler,
            loader_path=loader_path,
            plot_graphs=plot_graphs,
        )
        model_path = model_path.split("/")[-1]
        print(model_path)

        model = copy.deepcopy(_model)
        model_information.pop("swag_learning_rate")


# %% DEEP ENSEMBLES
if args.ensemble:
    models = []

    for i in range(5):
        models.append(
            build_model_function(
                dataset_train,
                gaussian=True,
                dropout_rate=args.dropout_rate,
                n_hidden=args.n_hidden,
            )
        )

    for index, model in enumerate(models):
        print(f"\x1b[1;31mEnsemble: Treino do modelo {index}\x1b[0m")

        learning_rate = 1e-4
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,  # , momentum=args.momentum, weight_decay=args.wd
            weight_decay=1e-3,
        )
        criterion = losses.gaussian_nll  # Precisa ser treinada com saída NLL

        # Treinando os 5 modelos do ensemble
        epochs = args.epochs

        trigger_time = 0
        train_loss_array, test_loss_array = [], []
        trigger_time = 0
        for epoch in range(epochs):
            optimizer, learning_rate = learning_schedule(optimizer, epoch)
            time_ep = time.time()
            train_function = (
                utils.train_epoch_adversarial
                if not (args.model in ["wtg", "nwp"])
                else utils.train_epoch
            )
            train_res = train_function(  # Não usa o criterion!
                loaders["train"],
                model,
                criterion,
                optimizer,
                regression=True,
                verbose=False,
                cuda=True,
                loader_scaler=loader_scaler,
            )
            test_res = utils.eval(  # Usa o criterion!
                loaders["test"],
                model,
                criterion,
                cuda=True,
                regression=True,
                loader_scaler=loader_scaler,
            )
            time_ep = time.time() - time_ep

            values = [
                epoch + 1,
                learning_rate,
                train_res["loss"],
                test_res["loss"],
                None,
                time_ep,
            ]

            test_loss_array.append(test_res["loss"])
            train_loss_array.append(train_res["loss"])

            print_tabulate(values, epoch)

            patience = 20
            if epoch >= 5:
                if test_loss_array[-1] > test_loss_array[-2]:
                    trigger_time += 1
                    if trigger_time >= patience:
                        print("Early stopping")
                        break
                else:
                    trigger_time = 0

    save_model(
        bayesian_framework="ensemble",
        models=models,
        loaders=loaders,
        model_information=model_information,
        loss_arrays=[train_loss_array, test_loss_array],
        calc_metrics=calc_metrics,
        dataset_test=dataset_test,
        dataset_path=DATASET_PATH,
        loader_scaler=loader_scaler,
        loader_path=loader_path,
        plot_graphs=plot_graphs,
    )

# %% NLL Baseline
if args.nllbaseline:
    model = build_model_function(
        dataset_train,
        gaussian=True,
        dropout_rate=args.dropout_rate,
        n_hidden=args.n_hidden,
    )

    print(f"\x1b[1;31mTreino do modelo NLL baseline")

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,  # , momentum=args.momentum, weight_decay=args.wd
        weight_decay=args.wd,
    )
    criterion = losses.gaussian_nll  # Precisa ser treinada com saída NLL

    epochs = args.epochs

    trigger_time = 0
    train_loss_array, test_loss_array = [], []
    for epoch in range(epochs):
        optimizer, learning_rate = learning_schedule(optimizer, epoch)
        time_ep = time.time()
        train_res = utils.train_epoch(
            loaders["train"],
            model,
            criterion,
            optimizer,
            regression=True,
            verbose=False,
            cuda=True,
            loader_scaler=loader_scaler,
        )
        test_res = utils.eval(
            loaders["test"],
            model,
            criterion,
            cuda=True,
            regression=True,
            loader_scaler=loader_scaler,
        )
        time_ep = time.time() - time_ep

        values = [
            epoch + 1,
            learning_rate,
            train_res["loss"],
            test_res["loss"],
            None,
            time_ep,
        ]

        test_loss_array.append(test_res["loss"])
        train_loss_array.append(train_res["loss"])

        print_tabulate(values, epoch)

        patience = 20
        if epoch >= 5:
            if test_loss_array[-1] > test_loss_array[-2]:
                trigger_time += 1
                if trigger_time >= patience:
                    print("Early stopping")
                    break

    save_model(
        bayesian_framework="nllbaseline",
        models=[model],
        loaders=loaders,
        model_information=model_information,
        loss_arrays=[train_loss_array, test_loss_array],
        calc_metrics=calc_metrics,
        dataset_test=dataset_test,
        dataset_path=DATASET_PATH,
        loader_scaler=loader_scaler,
        loader_path=loader_path,
        plot_graphs=plot_graphs,
    )
