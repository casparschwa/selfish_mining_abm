import numpy as np
import itertools
import time
import os
import logging

###############################################################################################
###################################### DEFINE PARAMETERS ######################################
###############################################################################################

# PARAMETERS TO LOOP OVER
alphas = np.linspace(0, 0.5, 3)
# 1 minute equals 60'000 milliseconds.
gammas = np.linspace(100, 500, 1) / 60000
repetitions = 1

# available topologogies: "UNIFORM", "ER", "BA"
topologies = ["UNIFORM", "ER", "BA"]

# available hashing power distributions: "UNIFORM", "POWERLAW", "EXPONENTIAL"
hash_distributions = ["UNIFORM", "POWERLAW", "EXPONENTIAL"]

# parameter list as input for multiprocessing
parameter_list = list(itertools.product(
    list(range(repetitions)), topologies, hash_distributions, gammas, alphas))

# SPECFIY TOPOLOGY
desired_avg_degree = 10  # applies to ER and RAND topology.
ba_m = 5  # relevant for BA topology; no. edges to attach from new node to existing nodes

# SPECIFY HASHING POWER DISTRIBUTION
pl_alpha = 1.88  # input parameter for powerlaw distribution
exp_lambda = 1  # input parameter for exponential distribution
# ADDITIONAL PARAMETERS
simulation_time = 100
number_of_nodes = 100
number_selfish_nodes = 1  # if there is more than 1 selfish miner, they act as "cartel"
number_honest_nodes = number_of_nodes - number_selfish_nodes
verbose = False

###############################################################################################
####################################### SET UP LOGGING ########################################
###############################################################################################
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
logging.info(f"Total number of repetitions: {repetitions}")
