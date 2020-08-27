import networkx as nx
import numpy as np
import pandas as pd
import os, time, logging
from tqdm import tqdm, trange
from blockchain import GillespieBlockchain


if __name__ == "__main__":

    ########################
    #### Set up logging ####
    timestamp = time.localtime()
    date_appendix = (
        str(timestamp[0])
        + "-"
        + str(timestamp[1])
        + "-"
        + str(timestamp[2])
        + "-"
        + str(timestamp[3])
        + ":"
        + str(timestamp[4])
        + ":"
        + str(timestamp[5])
    )
    fname = "blockchain_{}.log".format(date_appendix)
    path = os.path.join(os.getcwd(), "logs/{}".format(fname))
    logging.basicConfig(
        filename=path, level=logging.DEBUG, filemode="w", format="%(message)s",
    )
    ########################

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
        logging.info("number of nodes: {}".format(number_of_nodes))
        number_honest_nodes = number_of_nodes - number_selfish_nodes
        logging.info("number of honest nodes: {}".format(number_honest_nodes))
        total_hashing_power_honest = 1 - alpha
        logging.info(
            "Total honest hashing power: {}".format(total_hashing_power_honest)
        )
        if alpha == 0:
            hashing_power_selfish = np.zeros(number_selfish_nodes)
        else:
            hashing_power_selfish = np.random.random(number_selfish_nodes)
            hashing_power_selfish /= sum(hashing_power_selfish) / alpha
        hashing_power_honest = np.random.random(number_honest_nodes)
        hashing_power_honest /= sum(hashing_power_honest) / total_hashing_power_honest
        hashing_power = np.append(hashing_power_selfish, hashing_power_honest)
        is_selfish = np.append(
            np.ones(number_selfish_nodes), np.zeros(number_honest_nodes)
        )
        # randomize is_selfish and hashing_power arrays in unison
        randomize = np.arange(len(hashing_power))
        np.random.shuffle(randomize)

        # final hasing_power and is_selfish arrays
        hashing_power = hashing_power[randomize]
        logging.info("Hashing power array: {}".format(hashing_power))
        is_selfish = is_selfish[randomize]
        logging.info("Selfish nodes array: {}".format(is_selfish))

        return net_p2p, is_selfish, hashing_power

    ###################################
    #### PARAMETERS for simulation ####
    ###################################

    # TO SPECIFY
    number_of_nodes = 100
    number_selfish_nodes = 1
    logging.info("Number selfish nodes: {}".format(number_selfish_nodes))
    number_honest_nodes = number_of_nodes - number_selfish_nodes

    # total hashing power selfish nodes
    alphas = np.linspace(0, 0.5, 21)
    # alphas = np.array([0.2])

    # tau_nd is similar to gamma in original paper
    gammas = np.linspace(200, 1000, 3) / 60000  # 1 minute equals 60'000 milliseconds.
    # gammas = np.array([200 / 60000])

    # for random gnm graph
    number_of_neighbors = 2
    logging.info("Number of neighbors: {}".format(number_of_neighbors))

    # minutes in simulation world
    simulating_time = 1000

    # average results over how many repititons?
    repititions = 3
    logging.info("Repititions: {}".format(repititions))

    # log files?
    verbose = False

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
        "MSB",
        "MeanTimeHonestMainchainPropagation",
        "MediaTimeHonestMainchainPropagation",
        # "MinTimeHonestMainchainPropagation",
        # "MaxTimeHonestMainchainPropagation",
        # "MeanTimeFullyPropagated",
        # "MedianTimeFullyPropagated",
        # "MinTimeFullyPropagated",
        # "MaxTimeFullyPropagated",
        "MeanTimePropagation",
        "MedianTimePropagation",
        # "MinTimePropagation",
        # "MaxTimePropagation",
    ]

    # initialize data list with appropriate list of lists of 0's
    ## list_of_lists = [[sim.storage()],[sim.storage()],...,[sim.storage()]] -> list of (alpha * gamma) sim.storage() lists
    data_list = [[0] * len(columns) for i in range(len(alphas) * len(gammas))]

    start = time.time()

    for rep in trange(repititions, desc="Averaging loop", leave=False):

        ticker = 0

        for tau_nd in tqdm(gammas, desc="Gamma loop", leave=False):

            for alpha in tqdm(alphas, desc="Alpha loop", leave=False):

                if verbose:
                    logging.info(
                        "------------------------- NEW  RUN -------------------------\nRepition: {}; Gamma: {}; Alpha: {}".format(
                            rep, round(tau_nd, 3), round(alpha, 3)
                        )
                    )

                if alpha == 0:
                    model_setup = set_up_model(
                        number_of_nodes, 0, alpha, number_of_neighbors
                    )
                else:
                    model_setup = set_up_model(
                        number_of_nodes,
                        number_selfish_nodes,
                        alpha,
                        number_of_neighbors,
                    )

                model = GillespieBlockchain(
                    model_setup[0],
                    model_setup[1],
                    model_setup[2],
                    tau_nd,
                    verbose=verbose,
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

    runtime = time.time() - start

    print(
        "Simulation took {} minutes (in hours: {})".format(
            round(runtime / 60, 2), round(runtime / (60 * 60), 2)
        )
    )

    # average the data
    data_list = [[i / repititions for i in j] for j in data_list]

    # save data
    data = pd.DataFrame(data_list, columns=columns)
    filename = "data_{}.csv".format(date_appendix)
    path = os.path.join(str(os.getcwd()), "output/{}".format(filename))
    data.to_csv(path_or_buf=path)
