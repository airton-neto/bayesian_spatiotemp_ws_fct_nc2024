# %% Compõe as chamadas possíveis
from itertools import product

datasets = ["A"]  # "B"
models = ["mlp", "lstm", "convlstm"]
# bayesians = ["dropout", "swag", "multiswag", "ensemble", "nllbaseline"]
bayesians = [
    ["dropout", "swag", "multiswag"],
    ["ensemble", "nllbaseline"],
]
hiddens = [48]
rates = [0.2, 0.22, 0.24, 0.26, 0.28, 0.30]
swag_lrs = [
    "0.1"
]  # ["0.1,0.15,0.2,0.23,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9"]
grid_search = False

cmd = "python 2_BuildAllModels.py {} --model {} --dataset {} --n_hidden {} --dropout_rate {} --epochs 250"

for bayesian, model, dataset, hidden, rate in product(
    bayesians, models, datasets, hiddens, rates
):
    bayesian = " ".join([f"--{_bays}" for _bays in bayesian])
    if "multiswag" in bayesian:
        print(
            cmd.format(bayesian, model, dataset, hidden, rate)
            + " --swag_lrs {}".format(",".join(swag_lrs))
            + (" --grid_search --skip_plots" if grid_search else "")
        )
    else:
        print(cmd.format(bayesian, model, dataset, hidden, rate))

for dataset in datasets:
    print(
        f"python 2_BuildAllModels.py --model dummy --dataset {dataset} --epochs 1"
    )
