import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm, trange
from blockchain import GillespieBlockchain


def collect_results():
    pass


def set_up_model(number_of_nodes, number_selfish_nodes, alpha, number_of_neighbors):
    # generate network
    net_p2p = nx.gnm_random_graph(
        number_of_nodes, number_of_nodes * number_of_neighbors
    )
    # get largest connected component
    lcc_set = max(nx.connected_components(net_p2p), key=len)
    lcc_size = len(lcc_set)
    net_p2p = net_p2p.subgraph(lcc_set).copy()
    # some nodes may have been removed because they were not port of the lcc.
    # relabel nodes so that only nodes in lcc are labelled. (without it we run into problems where node labels are higher than the number of nodes -> loops run into index problems)
    net_p2p = nx.convert_node_labels_to_integers(net_p2p, first_label=0)

    # some nodes may have been removed as they were not part of the lcc -> update num nodes
    number_of_nodes = len(net_p2p)
    number_honest_nodes = number_of_nodes - number_selfish_nodes
    total_hashing_power_honest = 1 - alpha
    if alpha == 0:
        hashing_power_selfish = np.zeros(number_selfish_nodes)
    else:
        hashing_power_selfish = np.random.random(number_selfish_nodes)
        hashing_power_selfish /= sum(hashing_power_selfish) / alpha
    hashing_power_honest = np.random.random(number_honest_nodes)
    hashing_power_honest /= sum(hashing_power_honest) / total_hashing_power_honest
    hashing_power = np.append(hashing_power_selfish, hashing_power_honest)
    is_selfish = np.append(np.ones(number_selfish_nodes), np.zeros(number_honest_nodes))
    # randomize is_selfish and hashing_power arrays in unison
    randomize = np.arange(len(hashing_power))
    np.random.shuffle(randomize)

    # final hasing_power and is_selfish arrays
    hashing_power = hashing_power[randomize]
    is_selfish = is_selfish[randomize]

    return net_p2p, is_selfish, hashing_power


###################################
#### PARAMETERS for simulation ####
###################################

# TO SPECIFY
number_of_nodes = 200
number_selfish_nodes = 10
number_honest_nodes = number_of_nodes - number_selfish_nodes

# total hashing power selfish nodes
alphas = np.linspace(0, 0.5, 21)

# tau_nd is similar to gamma in original paper
gammas = np.array([0.001, 0.01, 0.1, 1])

# for random gnm graph
number_of_neighbors = 4

# minutes in simulation world
simulating_time = 1000

# average results over how many repititons?
repititions = 5


# DATA COLLECTION STUFF
columns = [
    "SimulationTime",
    "Alpha",
    "Gamma",
    "TotalBlocks",
    "SelfishBlocks",
    "HonestBlocks",
    "MainchainBlocks",
    "OrphanBlocks",
    "SelfishRevenue",
    "HonestRevenue",
    "RelativeSelfishRevenue",
    # "MeanTimeFullyPropagated",
    # "MedianTimeFullyPropagated",
    # "MinTimeFullyPropagated",
    # "MaxTimeFullyPropagated",
    "MeanTimePropagation",
    "MedianTimePropagation",
    "MinTimePropagation",
    "MaxTimePropagation",
]

# initialize data list with appropriate list of lists of 0's
## list_of_lists = [[sim.storage()],[sim.storage()],...,[sim.storage()]] -> list of (alpha * gamma) sim.storage() lists
data_list = [[0] * len(columns) for i in range(len(alphas) * len(gammas))]

for rep in trange(repititions, desc="Averaging loop", leave=False):

    ticker = 0

    for tau_nd in tqdm(gammas, desc="Gamma loop", leave=False):

        for alpha in tqdm(alphas, desc="Alpha loop", leave=False):

            if alpha == 0:
                model_setup = set_up_model(
                    number_of_nodes, 0, alpha, number_of_neighbors
                )
            else:
                model_setup = set_up_model(
                    number_of_nodes, number_selfish_nodes, alpha, number_of_neighbors
                )

            model = GillespieBlockchain(
                model_setup[0], model_setup[1], model_setup[2], tau_nd, verbose=False
            )
            while model.time < simulating_time:
                model.next_event()

            # get results and add exogenous information
            data_point = model.block_tree.results()
            exogenous_data = [simulating_time, alpha, tau_nd]
            for i in range(len(exogenous_data)):
                data_point.insert(i, exogenous_data[i])

            # add data points to aggregate data list
            for i in range(len(columns)):
                data_list[ticker][i] += data_point[i]

            ticker += 1

# average the data
data_list = [[i / repititions for i in j] for j in data_list]


data = pd.DataFrame(data_list, columns=columns)
data.to_csv("test_run.csv")
