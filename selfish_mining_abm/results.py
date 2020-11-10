import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from config import *

#####################
#### Data import ####
#####################

if use_import:
    fname = imported_data_filename
    data = pd.read_csv(filepath_or_buffer=path_import)
else:
    data = pd.read_csv(filepath_or_buffer=path)


####################################
#### Relative Pool Revenue Plot ####
####################################

def multi_plot(iterator1, iterator2):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(
        14, 12), sharex=True, sharey=True)
    marker_list = ["+", "x", "s", "o"]
    color_list = ["red", "green", "blue", "orange"]
    topology_names = ["Uniform random", "Erdos-Renyi", "Barabasi-Albert"]
    hash_distr_names = ["Uniform random", "Powerlaw", "Exponential"]
    unique_gammas = data["Gamma"].unique()

    header_is_topology = True if iterator1 == topologies else False

    for (ii, iter1) in enumerate(iterator1):
        for (jj, iter2) in enumerate(iterator2):

            # setup
            legend_label = f"Hashing Power Distribution: {hash_distr_names[jj]}" if header_is_topology else f"Topology: {topology_names[jj]}"
            title = f"Topology: {topology_names[ii]}" if header_is_topology else f"Hashing Power Distribution: {hash_distr_names[ii]}"
            title += f" | Latency: {np.round(unique_gammas[0],4)}"

            if header_is_topology:
                filt = data[
                    (data["Topology"] == iter1) &
                    (data["HashingPowerDistribution"] == iter2) &
                    (data["Gamma"] == unique_gammas[0])
                ]
            else:
                filt = data[
                    (data["Topology"] == iter2) &
                    (data["HashingPowerDistribution"] == iter1) &
                    (data["Gamma"] == unique_gammas[0])
                ]

            axs[ii].plot(
                filt["Alpha"],
                filt["RelativeSelfishRevenue"],
                label=legend_label,
                marker=marker_list[jj],
                color=color_list[jj],
                linestyle="-")
            axs[ii].set_xlabel(r"Relative Pool Size $\alpha$")
            axs[ii].set_ylabel("Relative Pool Revenue")

        axs[ii].plot(
            [0, 0.5],
            [0, 0.5],
            label="Honest Mining",
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        axs[ii].set_xlim(0, 0.5)
        axs[ii].set_ylim(0, 1)
        axs[ii].set_title(title)
        axs[ii].tick_params(direction="in")
        axs[ii].legend(loc='upper left')

    # save fig
    fig_filename = f"topologies_{fname[:-4]}.png" if header_is_topology else f"hash_distributions_{fname[:-4]}.png"
    path = os.path.dirname(os.getcwd()) + f"/figures/{fig_filename}"
    plt.savefig(path, bbox_inches="tight")


multi_plot(topologies, hash_distributions)
multi_plot(hash_distributions, topologies)

# # # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(
# # #     14, 12), sharex=True, sharey=True)
# # # marker_list = ["+", "x", "s", "o"]
# # # color_list = ["red", "green", "blue", "orange"]
# # # topology_names = ["Uniform random", "Erdos-Renyi", "Barabasi-Albert"]
# # # hash_distr_names = ["Uniform random", "Powerlaw", "Exponential"]

# # # for (ii, topology) in enumerate(topologies):
# # #     for (jj, hash_distr) in enumerate(hash_distributions):
# # #         filt = data[
# # #             (data["Topology"] == topology) &
# # #             (data["HashingPowerDistribution"] == hash_distr) &
# # #             (data["Gamma"] == gammas[0])
# # #         ]
# # #         axs[ii].plot(
# # #             filt["Alpha"],
# # #             filt["RelativeSelfishRevenue"],
# # #             label=f"Hashing Power Distribution: {hash_distr_names[jj]}",
# # #             marker=marker_list[jj],
# # #             color=color_list[jj],
# # #             linestyle="-")
# # #         axs[ii].set_xlabel(r"Relative Pool Size $\alpha$")
# # #         axs[ii].set_ylabel("Relative Pool Revenue")

# # #     axs[ii].plot(
# # #         [0, 0.5],
# # #         [0, 0.5],
# # #         label="Honest Mining",
# # #         color="black",
# # #         linestyle="--",
# # #         linewidth=1.0,
# # #     )
# # #     axs[ii].set_ylim(0, 1)
# # #     axs[ii].set_title(
# # #         f"Topology: {topology_names[ii]} | Latency: {np.round(gammas[0],4)}")
# # #     axs[ii].tick_params(direction="in")
# # #     axs[ii].legend(loc='upper left')

# # # # save fig
# # # fig1_filename = f"fig1_{fname[:-4]}.png"
# # # path = os.path.dirname(os.getcwd()) + f"/figures/{fig1_filename}"
# # # plt.savefig(path, bbox_inches="tight")

#################################
### Alpha Threshold Plot  #######
#################################


############################
# nonsensical in abm context
# def calc_threshold(gamma):
#     g = gamma
#     alpha_threshold_theory = (1 - g) / (3 - 2 * g)
#     return alpha_threshold_theory
############################

# # # unique_gammas = data["Gamma"].unique()
# # # alpha_thresholds_simulation = []

# # # for gamma in unique_gammas:
# # #     filt = (
# # #         (data["Gamma"] == gamma)
# # #         & (data["Alpha"] <= data["RelativeSelfishRevenue"])
# # #         & (data["Alpha"] != 0)
# # #     )
# # #     alpha_thresholds_simulation.append(min(list(data.loc[filt, "Alpha"])))

# # # fig2 = plt.figure()
# # # ax = fig2.add_subplot()
# # # ax = fig2.add_axes([0, 0, 1, 1])
# # # ax.plot(
# # #     unique_gammas,
# # #     alpha_thresholds_simulation,
# # #     color="black",
# # #     label="Simulation",
# # #     marker="s",
# # #     markerfacecolor="None",
# # #     linestyle="None",
# # # )
# # # ax.plot(
# # #     unique_gammas, calc_threshold(unique_gammas), color="black", label="Theory",
# # # )
# # # ax.set_xlabel(r"$\gamma$")
# # # ax.set_ylabel(r"Threshold $\alpha$")
# # # ax.tick_params(direction="in")
# # # ax.legend()

# # # # save fig
# # # fig2_filename = f"fig2_{fname[:-4]}.png"
# # # path = os.path.dirname(os.getcwd()) + f"/figures/{fig2_filename}"
# # # plt.savefig(path, bbox_inches="tight")


####################################
###########   MSB Model   ##########
####################################

# gammas = data["Gamma"].unique()
# marker_list = ["+", "x", "s", "o"]
# colour_list = ["red", "green", "blue", "orange"]

# fig3 = plt.figure()
# ax = fig3.add_subplot(1, 1, 1)
# ax = fig3.add_axes([0.0, 0.0, 1, 1])

# for (index, gamma) in enumerate(gammas):
#     # plotting only every 20th point -> [0::20]
#     # simulation values
#     ax.plot(
#         data[data["Gamma"] == gamma]["Alpha"],
#         data[data["Gamma"] == gamma]["SelfishMSB"],
#         label=r"$Avg. selfish miner - \gamma$ = {}".format(gamma),
#         # # # marker=marker_list[index],
#         # # # color=colour_list[index],
#         # # # markerfacecolor="None",
#         linestyle="-",
#     )
#     ax.plot(
#         data[data["Gamma"] == gamma]["Alpha"],
#         data[data["Gamma"] == gamma]["HonestMSB"],
#         label=r"$Avg. honest miner - \gamma$ = {}".format(gamma),
#         # # # marker=marker_list[index],
#         # # # color=colour_list[index],
#         # # # markerfacecolor="None",
#         linestyle="-",
#     )

# # add significance level (MSB=2) line
# ax.plot(
#     [0, 0.5], [2, 2], label=r"$MSB = 2$", color="black", linestyle="--", linewidth=1.0,
# )

# ax.set_xlabel(r"Relative Pool Size $\alpha$")
# ax.set_ylabel("MSB")
# ax.tick_params(direction="in")
# ax.legend()

# # save fig
# fig3_filename = f"fig3_{fname[:-4]}.png"
# path = os.path.dirname(os.getcwd()) + f"/figures/{fig3_filename}"
# plt.savefig(path, bbox_inches="tight")
