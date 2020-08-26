import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time

import simulations


#####################
#### Data import ####
#####################

# IMPORTED OR DATASET GENERATED LAST?
use_import = True

# gets the last created filename (which is the latest dataset)
search_dir = os.path.join(os.getcwd(), "output/")
os.chdir(search_dir)
files = filter(os.path.isfile, os.listdir(search_dir))
files = [os.path.join(search_dir, f) for f in files]  # add path to each file
files.sort(key=lambda x: os.path.getmtime(x))
path = files[::-1][0]
fname = os.path.basename(path)

imported_data_filename = "import1.csv"
path_import = os.path.join(os.getcwd(), "{}".format(imported_data_filename))

if use_import:
    data = pd.read_csv(filepath_or_buffer=path_import)
else:
    data = pd.read_csv(filepath_or_buffer=path)


def calc_threshold(gamma):
    g = gamma
    alpha_threshold_theory = (1 - g) / (3 - 2 * g)
    return alpha_threshold_theory


####################################
#### Relative Pool Revenue Plot ####
####################################

gammas = data["Gamma"].unique()
marker_list = ["+", "x", "s", "o"]
colour_list = ["red", "green", "blue", "orange"]

fig1 = plt.figure()
ax = fig1.add_subplot(1, 1, 1)
ax = fig1.add_axes([0.1, 0.1, 0.75, 0.75])

for index, gamma in enumerate(gammas):
    # plotting only every 20th point -> [0::20]
    # simulation values
    ax.plot(
        data[data["Gamma"] == gamma]["Alpha"],
        data[data["Gamma"] == gamma]["RelativeSelfishRevenue"],
        label=r"$\gamma$ = {}".format(gamma),
        # # # marker=marker_list[index],
        # # # color=colour_list[index],
        # # # markerfacecolor="None",
        linestyle="-",
    )

    # # # # plotting only every 20th point -> [0::20]
    # # # # theoretical values
    # # # ax.plot(
    # # #     data[data["Gamma"] == gamma]["Alpha"],
    # # #     calc_revenue(data[data["Gamma"] == gamma]["Alpha"], gamma),
    # # #     label=r"$\gamma$ = {} (theory)".format(gamma),
    # # #     # # # color=colour_list[index],
    # # #     # # # markerfacecolor="None",
    # # # )

ax.plot(
    [0, 0.5],
    [0, 0.5],
    label="Honest Mining",
    color="black",
    linestyle="--",
    linewidth=1.0,
)

ax.set_xlabel(r"Relative Pool Size $\alpha$")
ax.set_ylabel("Relative Pool Revenue")
ax.tick_params(direction="in")
ax.legend()

fig1_filename = "fig1_{}.png".format(fname)
path = os.path.join(os.getcwd(), "{}".format(fig1_filename))
plt.savefig(path)

#################################
### Alpha Threshold Plot  #######
#################################

unique_gammas = data["Gamma"].unique()
alpha_thresholds_simulation = []

for gamma in unique_gammas:
    filt = (
        (data["Gamma"] == gamma)
        & (data["Alpha"] <= data["RelativeSelfishRevenue"])
        & (data["Alpha"] != 0)
    )
    alpha_thresholds_simulation.append(min(list(data.loc[filt, "Alpha"])))

fig2 = plt.figure()
ax = fig2.add_subplot()
ax = fig2.add_axes([0, 0, 1, 1])
ax.plot(
    unique_gammas,
    alpha_thresholds_simulation,
    color="black",
    label="Simulation",
    marker="s",
    markerfacecolor="None",
    linestyle="None",
)
ax.plot(
    unique_gammas, calc_threshold(unique_gammas), color="black", label="Theory",
)
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"Threshold $\alpha$")
ax.tick_params(direction="in")
ax.legend()

# plt.tight_layout()
fig2_filename = "fig2_{}.png".format(fname)
path = os.path.join(os.getcwd(), "{}".format(fig2_filename))
plt.savefig(path)
plt.show()


####################################
###########   MSB Model   ##########
####################################

gammas = data["Gamma"].unique()
marker_list = ["+", "x", "s", "o"]
colour_list = ["red", "green", "blue", "orange"]

fig3 = plt.figure()
ax = fig3.add_subplot(1, 1, 1)
ax = fig3.add_axes([0.0, 0.0, 1, 1])

for index, gamma in enumerate(gammas):
    # plotting only every 20th point -> [0::20]
    # simulation values
    ax.plot(
        data[data["Gamma"] == gamma]["Alpha"],
        data[data["Gamma"] == gamma]["MSB"],
        label=r"$\gamma$ = {}".format(gamma),
        # # # marker=marker_list[index],
        # # # color=colour_list[index],
        # # # markerfacecolor="None",
        linestyle="-",
    )

ax.set_xlabel(r"Relative Pool Size $\alpha$")
ax.set_ylabel("MSB")
ax.tick_params(direction="in")
ax.legend()
plt.show()


fig3_filename = "fig3_{}.png".format(fname)
path = os.path.join(os.getcwd(), "{}".format(fig3_filename))
plt.savefig(path)
