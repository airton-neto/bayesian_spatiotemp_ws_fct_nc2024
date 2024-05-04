# %%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

quantiles = list(np.arange(0.05, 1.05, 0.05))
ideal = quantiles
main_quantiles = list(np.arange(0.1, 1, 0.1))

# %% usadp pra fzr um exemplo
perturbacao = np.sin(2 * np.pi * np.arange(0.05, 1.05, 0.05))
perturbacao[0] = 0
perturbacao[-1] = 0
cumulatives = np.flip(np.array(quantiles + perturbacao / 8))
cumulatives_best = np.flip(np.array(quantiles + perturbacao / 32))

cumulatives = cumulatives / max(cumulatives)
cumulatives_best = cumulatives_best / max(cumulatives_best)

# %%
fig, ax = plt.subplots(figsize=[5, 5])
ax.plot(quantiles, ideal, linestyle="--", color="#bdbdbd", linewidth=3)

realizado = np.flip(cumulatives)
ax.plot(
    quantiles,
    realizado,
    linestyle="--",
    # marker="o",
    color="#5c1010",
)

realizado_best = np.flip(cumulatives_best)
ax.plot(
    quantiles,
    realizado_best,
    linestyle="--",
    # marker="o",
    color="red",
)

ax.set_xticks(main_quantiles)
ax.set_ylabel("Empirical")
ax.set_xlabel("Nominal")


custom_lines = [
    Line2D([], [], color="red", lw=2),
    Line2D([], [], color="#5c1010", lw=2),
    Line2D([], [], color="#bdbdbd", lw=2),
]
ax.legend(
    custom_lines,
    ["Well-Calibrated", "Poorly-Calibrated", "Ideal"],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    # loc=0,
    labelspacing=2,
    columnspacing=1.5,
    frameon=False,
    # fontsize=8,
    ncol=3,
)

# %%
fig.savefig("../Figuras/talagrad_cumulatve_example.png")

###################################
# %% Dierpsao Erros Example
import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=[5, 5])

real = np.arange(-14, 16, 2)
predito = real + np.random.normal(0, 1, size=len(real))
predito_ruim = real + np.random.normal(-1, 5, size=len(real))

ax.scatter(real, predito, color="blue")
ax.scatter(real, predito_ruim, color="red")
ax.plot(
    list(range(-14, 16, 2)),
    list(range(-14, 16, 2)),
    linestyle="--",
    color="#bdbdbd",
    linewidth=2,
)

ax.set_xlim([-14, 14])
ax.set_ylim([-14, 14])


ax.set_yticks([-10, -5, 0, 5, 10])
ax.set_xticks([-10, -5, 0, 5, 10])

ax.set_ylabel("Observed Values")
ax.set_xlabel("Predicted Values")

fig.savefig("../Figuras/errorplot_example.png")
