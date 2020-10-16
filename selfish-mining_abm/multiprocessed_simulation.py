import networkx as nx
import numpy as np
import pandas as pd
import powerlaw as pl
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

    # PARAMETERS TO LOOP OVER
    alphas = np.linspace(0, 0.5, 11)
    # 1 minute equals 60'000 milliseconds.
    gammas = np.linspace(100, 500, 1) / 60000
    repititions = 2
    # parameter list as input for multiprocessing
    parameter_list = list(itertools.product(
        list(range(repititions)), gammas, alphas))

    # SPECFIY TOPOLOGY
    # "UNIFORM", "ER" (Erdos-Renyi) or "BA" (Barabasi-Albert)
    topology = "BA"
    desired_avg_degree = 10  # applies to ER and RAND topology.
    ba_m = 5  # relevant for BA topology; no. edges to attach from new node to existing nodes

    # SPECIFY HASHING POWER DISTRIBUTION
    # "UNIFORM", "POWERLAW", "EXPONENTIAL"
    hash_distribution = "EXPONENTIAl"
    pl_alpha = 1.88  # input parameter for powerlaw distribution
    exp_lambda = 1  # input parameter for exponential distribution

    # ADDITIONAL PARAMETERS
    simulation_time = 100
    number_of_nodes = 100
    number_selfish_nodes = 1  # if there is more than 1 selfish miner, they act as "cartel"
    number_honest_nodes = number_of_nodes - number_selfish_nodes
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
    # log basic information
    logging.info(f"List of Alpha values to iterate over: {alphas}")
    logging.info(f"List of Gamma values to iterate over: {gammas}")
    logging.info(
        f"Desired average degree for ER and RAND graph: {desired_avg_degree}")
    logging.info(f"Total simulation time: {simulation_time}")
    logging.info(f"Total number of repititions: {repititions}")

    ##############################################################################
    #### DEFINE FUNCTIONS TO SET UP & RUN (SINGLE) MODEL #########################
    ##############################################################################

    def __set_up_topology(topology, number_of_nodes, desired_avg_degree, ba_m):
        # generate network depending on topology parameter
        if topology == "UNIFORM":
            net_p2p = nx.gnm_random_graph(
                number_of_nodes, number_of_nodes * desired_avg_degree / 2)

        elif topology == "ER":
            p = desired_avg_degree / number_of_nodes
            net_p2p = nx.fast_gnp_random_graph(number_of_nodes, p)

        elif topology == "BA":
            net_p2p = nx.barabasi_albert_graph(number_of_nodes, ba_m)

        # log actually realized average degree
        logging.info(
            f"Actual avg. degree: {2*net_p2p.number_of_edges()/ net_p2p.number_of_nodes()}")

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

    def __set_up_hash_distr(number_selfish_nodes, number_honest_nodes, alpha):
        # make sure that when there are no selfish nodes that alpha is never unequal 0. (in case you want to simulate only honest nodes)
        assert not (number_selfish_nodes == 0 and alpha != 0)

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
        hashing_power = np.append(hashing_power_selfish, hashing_power_honest)

        # create an is_selfish vector that corresponds to the order of the hashing_power vector
        is_selfish = np.append(np.ones(number_selfish_nodes),
                               np.zeros(number_honest_nodes))

        # finally, randomize is_selfish and hashing_power arrays in unison
        randomize = np.arange(len(hashing_power))
        np.random.shuffle(randomize)
        hashing_power = hashing_power[randomize]
        is_selfish = is_selfish[randomize]

        return hashing_power, is_selfish

    def __set_up_model(
        topology, number_of_nodes, number_selfish_nodes, alpha, desired_avg_degree, ba_m
    ):
        net_p2p, number_honest_nodes = __set_up_topology(
            topology, number_of_nodes, desired_avg_degree, ba_m)

        hashing_power, is_selfish = __set_up_hash_distr(
            number_selfish_nodes, number_honest_nodes, alpha)

        return net_p2p, hashing_power, is_selfish

    # function that handles a single run for a given set of parameters
    def run_simulation(parameters):

        repitition, gamma, alpha = parameters[0], parameters[1], parameters[2]

        if alpha == 0:
            model_setup = __set_up_model(
                topology, number_of_nodes, number_selfish_nodes, 0, desired_avg_degree, ba_m
            )
        else:
            model_setup = __set_up_model(
                topology, number_of_nodes, number_selfish_nodes, alpha, desired_avg_degree, ba_m
            )

        model = GillespieBlockchain(
            model_setup[0], model_setup[1], model_setup[2], gamma, verbose=verbose
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

    # dump results into list -> res is list of lists with the data_points
    res = []
    for result in results:
        res.append(result)

    # create a list of lists of lists, where the inner list of lists are lists of the results with the identical parameters alpha and gamma
    res = [res[i:: (len(res) // repititions)]
           for i in range(len(res) // repititions)]

    # data_list averages the result of res's inner list of lists (i.e. results of identical parameter setup)
    data_list = []
    for i in res:
        result = [sum(j) for j in (zip(*i))]
        # average data
        data_list.append([i / repititions for i in result])

    # create dataframe
    data = pd.DataFrame(data_list, columns=columns)

    # save data using date appendix from logging setup
    filename = f"data_{date_appendix}.csv"
    path = parent_dir + f"/output/data/{filename}"
    data.to_csv(path_or_buf=path)
