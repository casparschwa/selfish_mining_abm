import blockchain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import itertools
import random
import sys
import powerlaw as pl
import blockchain
import time

start = time.perf_counter()

############
## CONFIG ##
############

alphas = [0.2, 0.4]
gammas = [0.14, 1.58, 17.78]
parameter_list = list(itertools.product(alphas, gammas))
print("Alpha values: ", alphas)
print("Gamma values (tau): ", gammas)

centrality_measure = "RANDOM"
topology = "UNIFORM"
hash_distribution = "UNIFORM"
simulation_time = 2000000
print("Simulation time: ", simulation_time)
number_of_nodes = 120
number_selfish_nodes = 20  # if there is more than 1 selfish miner, they act as "cartel"
number_honest_nodes = number_of_nodes - number_selfish_nodes
verbose = False

# # SPECFIY TOPOLOGY
desired_avg_degree = 10  # applies to ER and RAND topology.
ba_m = 5  # relevant for BA topology; no. edges to attach from new node to existing nodes

# SPECIFY HASHING POWER DISTRIBUTION
pl_alpha = 1.88  # input parameter for powerlaw distribution
exp_lambda = 1  # input parameter for exponential distribution
# ADDITIONAL PARAMETERS


def set_up_topology(topology, number_of_nodes, desired_avg_degree, ba_m):
    # generate network depending on topology parameter
    if topology == "UNIFORM":
        net_p2p = nx.random_degree_sequence_graph(
            [desired_avg_degree for i in range(number_of_nodes)])

    elif topology == "ER":
        p = desired_avg_degree / number_of_nodes
        net_p2p = nx.fast_gnp_random_graph(number_of_nodes, p)

    elif topology == "BA":
        net_p2p = nx.barabasi_albert_graph(number_of_nodes, ba_m)

    # get largest connected component
    lcc_set = max(nx.connected_components(net_p2p), key=len)
    net_p2p = net_p2p.subgraph(lcc_set).copy()
    # some nodes may have been removed because they were not port of the lcc.
    # relabel nodes so that only nodes in lcc are labelled. (without it we run into problems where node labels are higher than the number of nodes -> loops run into indexing problems)
    net_p2p = nx.convert_node_labels_to_integers(
        net_p2p, first_label=0)

    # some nodes may have been removed as they were not part of the lcc -> update num nodes
    number_of_nodes = len(net_p2p)
    # (brute-forcing number of selfish nodes to stay unchanged)
    number_honest_nodes = number_of_nodes - number_selfish_nodes

    return net_p2p, number_honest_nodes


def set_up_hash_distr(net_p2p, centrality_measure, hash_distribution, number_selfish_nodes, number_honest_nodes, alpha):
    # make sure that when there are no selfish nodes that alpha is never unequal 0. (in case you want to simulate only honest nodes)
    assert not (number_selfish_nodes == 0 and alpha !=
                0), "Alpha unequal 0 with no selfish nodes"

    if hash_distribution == "UNIFORM":
        hashing_power_selfish = np.random.random(number_selfish_nodes)
        hashing_power_honest = np.random.random(number_honest_nodes)

    elif hash_distribution == "POWERLAW":
        power_distrib = pl.Power_Law(parameters=[pl_alpha], discrete=False)
        hashing_power_selfish = power_distrib.generate_random(
            number_selfish_nodes)
        hashing_power_honest = power_distrib.generate_random(
            number_honest_nodes)

    elif hash_distribution == "EXPONENTIAL":
        exp_distrib = pl.Exponential(parameters=[exp_lambda])
        hashing_power_selfish = exp_distrib.generate_random(
            number_selfish_nodes)
        hashing_power_honest = exp_distrib.generate_random(
            number_honest_nodes)

    # normalize vector so that sum of selfish hashing power equals alpha & honest hashing power equals 1-alpha.
    if number_selfish_nodes != 0:
        hashing_power_selfish /= sum(hashing_power_selfish)
        hashing_power_selfish *= alpha
    hashing_power_honest /= sum(hashing_power_honest) / (1 - alpha)

    # combine selfish and honest hashing power vectors together
    hashing_power_unsorted = np.append(
        hashing_power_selfish, hashing_power_honest)

    if centrality_measure == "RANDOM":
        # create an is_selfish vector that corresponds to the order of the hashing_power vector
        is_selfish = np.append(np.ones(number_selfish_nodes),
                               np.zeros(number_honest_nodes))

        # finally, randomize is_selfish and hashing_power arrays in unison
        randomize = np.arange(len(hashing_power_unsorted))
        np.random.shuffle(randomize)
        hashing_power = hashing_power_unsorted[randomize]
        is_selfish = is_selfish[randomize]

    elif centrality_measure == "BETWEENNESS":
        # compute betweenness centrality and sort it
        btwn = nx.betweenness_centrality(net_p2p)
        btwn_sorted = {k: v for k, v in sorted(
            btwn.items(), key=lambda item: item[1], reverse=True)}
        # return node indeces sorted for betweenness centrality
        btwn_sorted_indices = list(btwn_sorted.keys())

        selfish_indices = list(btwn_sorted.keys())[:number_selfish_nodes]
        honest_indices = list(btwn_sorted.keys())[
            number_selfish_nodes:len(btwn)]

        # set selifsh nodes according to betweenness centrality
        is_selfish = np.zeros(number_honest_nodes+number_selfish_nodes)
        for i in selfish_indices:
            is_selfish[i] = 1

        # sort hashing power vector so that selfish nodes are assigned correct hashing power
        hashing_power = hashing_power_unsorted.copy()
        for (index, value) in enumerate(btwn_sorted):
            hashing_power[value] = hashing_power_unsorted[index]

    return hashing_power, is_selfish


def set_up_model(
    centrality_measure, topology, hash_distribution, number_of_nodes, number_selfish_nodes, alpha, desired_avg_degree, ba_m
):
    net_p2p, number_honest_nodes = set_up_topology(
        topology, number_of_nodes, desired_avg_degree, ba_m)

    hashing_power, is_selfish = set_up_hash_distr(
        net_p2p, centrality_measure, hash_distribution, number_selfish_nodes, number_honest_nodes, alpha)

    return net_p2p, hashing_power, is_selfish

############
# SIMULATE #
############


def simulate(centrality_measure, topology, hash_distribution, number_of_nodes, number_selfish_nodes, alpha, desired_avg_degree, ba_m, gamma):
    if alpha == 0:
        net_p2p, hashing_power, is_selfish = set_up_model(
            centrality_measure, topology, hash_distribution, number_of_nodes, number_selfish_nodes, 0, desired_avg_degree, ba_m
        )
    else:
        net_p2p, hashing_power, is_selfish = set_up_model(
            centrality_measure, topology, hash_distribution, number_of_nodes, number_selfish_nodes, alpha, desired_avg_degree, ba_m
        )

    model = blockchain.GillespieBlockchain(
        net_p2p, is_selfish, hashing_power, gamma, verbose=verbose
    )

    while model.time < simulation_time:
        model.next_event()
    model.block_tree.tag_main_chain()
    return model


def get_scatter_data(model):

    node_ids = []
    for node in model.nodes:
        node_ids.append(node.id)

    is_selfish = list(model.is_selfish)
    hashing_power = model.hashing_power

    num_blocks_mined_by = [0] * len(node_ids)  # all blocks
    num_mc_blocks_mined_by = [0] * len(node_ids)  # main chain
    num_oc_blocks_mined_by = [0] * len(node_ids)  # off chain

    for block in model.block_tree.tree.nodes:
        if model.block_tree.attributes[block]["miner"] == "genesis":
            continue
        miner = model.block_tree[block]["miner"]
        if model.block_tree.attributes[block]["main_chain"]:
            num_mc_blocks_mined_by[miner] += 1
        else:
            num_oc_blocks_mined_by[miner] += 1
        num_blocks_mined_by[miner] += 1

    data_list = [node_ids, is_selfish, hashing_power,
                 num_blocks_mined_by, num_mc_blocks_mined_by, num_oc_blocks_mined_by]

    columns = ["node", "selfish", "hashing power",
               "total blocks", "mainchain blocks", "orphan blocks"]

    df = pd.DataFrame(columns=columns)
    for (index, column) in enumerate(columns):
        df[column] = data_list[index]
        df["alpha"] = alpha
        df["gamma"] = gamma
        df["time"] = simulation_time

    parent_dir = os.path.dirname(os.getcwd())
    filename = f"scatterplot_alpha_{alpha}_gamma_{gamma}.csv"
    path = parent_dir + \
        f"/output/scatter_data/{filename}"

    df.to_csv(path_or_buf=path)


for parameters in parameter_list:
    alpha, gamma = parameters
    model = simulate(centrality_measure, topology, hash_distribution,
                     number_of_nodes, number_selfish_nodes, alpha, desired_avg_degree, ba_m, gamma)
    get_scatter_data(model)

finish = time.perf_counter()
print(
    f"Simulation took {round((finish-start)/60,2)} minutes (in hours: {round((finish-start)/(60*60),2)})")
