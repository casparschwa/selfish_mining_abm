import networkx as nx
import numpy as np
import pandas as pd
import powerlaw as pl
import itertools
import concurrent.futures
from tqdm import tqdm
import os
import time
import logging
from config import *
from blockchain import GillespieBlockchain


if __name__ == "__main__":

    ##############################################################################
    #### DEFINE FUNCTIONS TO SET UP & RUN (SINGLE) MODEL #########################
    ##############################################################################

    def __set_up_topology(topology, number_of_nodes, desired_avg_degree, ba_m):
        # generate network depending on topology parameter
        if topology == "UNIFORM":
            net_p2p = nx.random_degree_sequence_graph(
                [desired_avg_degree for i in range(number_of_nodes)])

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

    def __set_up_hash_distr(net_p2p, centrality_measure, hash_distribution, number_selfish_nodes, number_honest_nodes, alpha):
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

        elif centrality_measure == "BETWEENNESS" or centrality_measure == "HASHING":
            # compute betweenness centrality and sort it
            btwn = nx.betweenness_centrality(net_p2p)
            btwn_sorted = {k: v for k, v in sorted(
                btwn.items(), key=lambda item: item[1], reverse=True)}
            # return node indeces sorted for betweenness centrality
            btwn_sorted_indices = list(btwn_sorted.keys())

            # generate is_selfish vector that ensures highest btwn ranked nodes are selfish
            # for the number of selfish nodes get the node id's of the highest btwn ranked nodes.
            selfish_indices = list(btwn_sorted.keys())[:number_selfish_nodes]
            is_selfish = np.zeros(number_honest_nodes+number_selfish_nodes)
            for i in selfish_indices:
                is_selfish[i] = 1

            if centrality_measure == "BETWEENNESS":
                # sort hashing power vector so that selfish nodes are assigned correct hashing power
                hashing_power = hashing_power_unsorted.copy()
                for (index, value) in enumerate(btwn_sorted):
                    hashing_power[value] = hashing_power_unsorted[index]

            elif centrality_measure == "HASHING":
                # sort hashing power vector but separately for selfish and honest nodes and attach
                hashing_power_sorted = np.append(
                    sorted(hashing_power_selfish, reverse=True),
                    sorted(hashing_power_honest, reverse=True)
                )
                # sort hashing power vector so that selfish nodes have highest btwn ranked nodes and:
                # selfish nodes that have more hashing power are assigned higher btwn ranked nodes
                # honest nodes that have more hashing power are assigned higher btwn ranked nodes
                hashing_power = hashing_power_sorted.copy()
                for (index, value) in enumerate(btwn_sorted):
                    hashing_power[value] = hashing_power_sorted[index]

        return hashing_power, is_selfish

    def __set_up_model(
        centrality_measure, topology, hash_distribution, number_of_nodes, number_selfish_nodes, alpha, desired_avg_degree, ba_m
    ):
        net_p2p, number_honest_nodes = __set_up_topology(
            topology, number_of_nodes, desired_avg_degree, ba_m)

        hashing_power, is_selfish = __set_up_hash_distr(
            net_p2p, centrality_measure, hash_distribution, number_selfish_nodes, number_honest_nodes, alpha)

        return net_p2p, hashing_power, is_selfish

    # function that handles a single run for a given set of parameters
    def run_simulation(parameters):

        # unpack parameters list
        repetition, centrality_measure, topology, hash_distribution, gamma, alpha = parameters

        if alpha == 0:
            net_p2p, hashing_power, is_selfish = __set_up_model(
                centrality_measure, topology, hash_distribution, number_of_nodes, number_selfish_nodes, 0, desired_avg_degree, ba_m
            )
        else:
            net_p2p, hashing_power, is_selfish = __set_up_model(
                centrality_measure, topology, hash_distribution, number_of_nodes, number_selfish_nodes, alpha, desired_avg_degree, ba_m
            )

        model = GillespieBlockchain(
            net_p2p, is_selfish, hashing_power, gamma, verbose=verbose
        )

        while model.time < simulation_time:
            model.next_event()

        # get results
        data_point = model.block_tree.results()
        exogenous_data = [simulation_time, centrality_measure, topology,
                          hash_distribution, alpha, gamma]
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
        "CentralityMeasure",
        "Topology",
        "HashingPowerDistribution",
        "Alpha",
        "Gamma",
        "TotalBlocks",
        "SelfishBlocks",
        "HonestBlocks",
        "MainchainBlocks",
        "OrphanBlocks",
        "MainchainBlockRate",
        "OrphanBlockRate",
        "SelfishRevenue",
        "HonestRevenue",
        "RelativeSelfishRevenue",
        "SelfishMSB",
        "HonestMSB",
        "MeanTimeHonestMainchainPropagation",
        "MedianTimeHonestMainchainPropagation",
        "MeanTimePropagation",
        "MedianTimePropagation",
        "GiniHashrate",
        "GiniMainchain",
        "GiniOffchain",
        "GiniBoth",
        "UniqueMinersMainchain",
        # "MinTimeHonestMainchainPropagation",
        # "MaxTimeHonestMainchainPropagation",
        # "MeanTimeFullyPropagated",
        # "MedianTimeFullyPropagated",
        # "MinTimeFullyPropagated",
        # "MaxTimeFullyPropagated",
        # "MinTimePropagation",
        # "MaxTimePropagation",
    ]

    # create a list of lists of lists, where the inner list of lists are lists of the results with the
    # identical parameters alpha and gamma
    res = [results[i:: (len(results) // repetitions)]
           for i in range(len(results) // repetitions)]

    # data_list averages the result of res's inner list of lists (i.e. results of identical parameter setup)
    data_list = []
    for i in res:
        ''' Explanation of list concatenation below
        zip(*i) gets values element-wise so that we can just sum and average them by dividing them by 
        number of repetitions. However, we also have strings describing the topology and hash distribution,
        so we need to make sure we only do calculations when it's not a string. This is handled by 
        "if no isinstance...". If it's a string, we just take the first string of the tuple (they are all 
        the same)
        '''
        result = [sum(j) / repetitions if not isinstance(j[0], str)
                  else j[0] for j in (zip(*i))]
        data_list.append(result)

    # create dataframe
    data = pd.DataFrame(data_list, columns=columns)

    # save data using date appendix from logging setup
    filename = f"data_{date_appendix}.csv"
    path = parent_dir + f"/output/data/{filename}"
    data.to_csv(path_or_buf=path)
