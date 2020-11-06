import networkx as nx
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import os
import time
import logging
import itertools
from blockchain import GillespieBlockchain


if __name__ == "__main__":

    ##############################################################################
    #### DEFINE PARAMETERS #######################################################
    ##############################################################################

    # parameters to loop over
    alphas = np.linspace(0, 0.5, 3)
    # 1 minute equals 60'000 milliseconds.
    gammas = np.linspace(100, 1000, 1) / 60000
    repetitions = 2
    # parameter list as input for multiprocessing
    parameter_list = list(itertools.product(
        list(range(repetitions)), gammas, alphas))

    # additional parameters
    simulation_time = 100
    number_of_nodes = 100
    number_selfish_nodes = 1
    number_honest_nodes = number_of_nodes - number_selfish_nodes
    number_of_neighbors = 2  # input required for random graph
    verbose = False

    ##############################################################################
    #### SET UP LOGGING ##########################################################
    ##############################################################################

    timestamp = time.localtime()
    date_appendix = (
        str(timestamp[1])
        + "-"
        + str(timestamp[2])
        + "-"
        + str(timestamp[3])
        + "_"
        + str(timestamp[4])
        + "_"
        + str(timestamp[5])
    )
    fname = f"{date_appendix}.log"
    parent_dir = os.path.dirname(os.getcwd())
    path = parent_dir + f"/output/logs/{fname}"
    logging.basicConfig(
        filename=path, level=logging.DEBUG, filemode="w", format="%(message)s",
    )
    # log some basic information
    logging.info(f"List of Alpha values to iterate over: {alphas}")
    logging.info(f"List of Gamma values to iterate over: {gammas}")
    logging.info(f"Number of neighbors in random graph: {number_of_neighbors}")
    logging.info(f"Total simulation time: {simulation_time}")
    logging.info(f"Total number of repetitions: {repetitions}")

    ##############################################################################
    #### DEFINE FUNCTIONS TO SET UP & RUN (SINGLE) MODEL #########################
    ##############################################################################

    # function to set up the model parameters
    def __set_up_model(
        number_of_nodes, number_selfish_nodes, alpha, number_of_neighbors
    ):
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
        hashing_power_honest /= sum(hashing_power_honest) / \
            total_hashing_power_honest
        hashing_power = np.append(hashing_power_selfish, hashing_power_honest)
        is_selfish = np.append(
            np.ones(number_selfish_nodes), np.zeros(number_honest_nodes)
        )
        # randomize is_selfish and hashing_power arrays in unison
        randomize = np.arange(len(hashing_power))
        np.random.shuffle(randomize)
        # final hasing_power and is_selfish arrays
        hashing_power = hashing_power[randomize]
        is_selfish = is_selfish[randomize]
        return net_p2p, is_selfish, hashing_power

    # function that handles a single run for a given set of parameters
    def run_simulation(parameters):

        repetition = parameters[0]
        gamma = parameters[1]
        alpha = parameters[2]

        if alpha == 0:
            model_setup = __set_up_model(
                number_of_nodes, number_selfish_nodes, 0, number_of_neighbors
            )
        else:
            model_setup = __set_up_model(
                number_of_nodes, number_selfish_nodes, alpha, number_of_neighbors
            )

        model = GillespieBlockchain(
            model_setup[0], model_setup[1], model_setup[2], gamma, tau_mine=1, verbose=verbose
        )

        while model.time < simulation_time:
            model.next_event()

        # get results
        data_point = model.block_tree.results()
        exogenous_data = [simulation_time, alpha, gamma]
        # add exogenous data to data_point
        for i in range(len(exogenous_data)):
            data_point.insert(i, exogenous_data[i])

        return data_point

    ##############################################################################
    #### RUN MULTIPROCESS SIMULATION #############################################
    ##############################################################################

    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(run_simulation, parameter_list), total=len(parameter_list)
            )
        )

    finish = time.perf_counter()
    print(
        f"Simulation took {round((finish-start)/60,2)} minutes (in hours: {round((finish-start)/(60*60),2)})"
    )

    ##############################################################################
    #### HANDLE & SAVE DATA ######################################################
    ##############################################################################

    # columns in data set
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
        "SelfishMSB",
        "HonestMSB",
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

    # create a list of lists of lists, where the inner list of lists are lists of the results with the identical parameters alpha and gamma
    res = [results[i:: (len(results) // repetitions)]
           for i in range(len(results) // repetitions)]

    logging.info(f"\nresults (length: {len(results)}):\n{results}\n \n")
    logging.info(f"res (length: {len(res)}):\n{res}\n \n")
    print(results, "\n")
    print(res)

    # data_list averages the result of res's inner list of lists (i.e. results of identical parameter setup)
    data_list = []
    for i in res:
        result = [sum(j) for j in (zip(*i))]
        # average data
        data_list.append([i / repetitions for i in result])

    # create dataframe
    data = pd.DataFrame(data_list, columns=columns)

    # save data using date appendix from logging setup
    filename = f"data_{date_appendix}.csv"
    path = parent_dir + f"/output/data/{filename}"
    data.to_csv(path_or_buf=path)
