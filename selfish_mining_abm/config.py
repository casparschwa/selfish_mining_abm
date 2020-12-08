import numpy as np
import itertools
import time
import os
import logging

###############################################################################################
###################################### DEFINE PARAMETERS ######################################
###############################################################################################

# PARAMETERS TO LOOP OVER
alphas = np.linspace(0, 0.5, 21)
# 1 minute equals 60'000 milliseconds.
gammas = np.logspace(np.log10(0.0001), np.log10(1000), 21)
# number of repetitions to average results over
repetitions = 20

# Hand/pick selfish miners: "RANDOM", "BETWEENNESS"
centrality_measures = ["RANDOM", "BETWEENNESS", "HASHING"]

# available topologogies: "UNIFORM", "ER", "BA"
topologies = ["UNIFORM", "ER", "BA"]

# available hashing power distributions: "UNIFORM", "POWERLAW", "EXPONENTIAL"
hash_distributions = ["UNIFORM", "POWERLAW", "EXPONENTIAL"]

# parameter list as input for multiprocessing
parameter_list = list(itertools.product(
    list(range(repetitions)), centrality_measures, topologies, hash_distributions, gammas, alphas))

# SPECFIY TOPOLOGY
desired_avg_degree = 10  # applies to ER and RAND topology.
ba_m = 5  # relevant for BA topology; no. edges to attach from new node to existing nodes

# SPECIFY HASHING POWER DISTRIBUTION
pl_alpha = 2  # input parameter for powerlaw distribution
exp_lambda = 1  # input parameter for exponential distribution
# ADDITIONAL PARAMETERS
simulation_time = 10000
number_of_nodes = 100
number_selfish_nodes = 1  # if there is more than 1 selfish miner, they act as "cartel"
number_honest_nodes = number_of_nodes - number_selfish_nodes
verbose = False

#### MULTIPROCESSING PARAMETER ####
# define number of max processes. This allows to reduce workload for the machine to make it usable for other things besides simulation...
max_processes = 48

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
# log parameter setup
logging.info(f"List of Alpha values to iterate over: {alphas}")
logging.info(f"List of Gamma values to iterate over: {gammas}\n")
logging.info(f"Total number of repetitions: {repetitions}")
logging.info(
    f"List of centrality measures to iterate over: {centrality_measures}")
logging.info(f"List of topologies to iterate over: {topologies}")
logging.info(
    f"List of hash distributions to iterate over: {hash_distributions}\n")
logging.info(
    f"Desired average degree for ER and RAND graph: {desired_avg_degree}")
logging.info(f"Barabasi-Albert graph, m: {ba_m}")
logging.info(f"Power law hash distribution, alpha: {pl_alpha}")
logging.info(f"Exponential hash distribution, lambda: {exp_lambda}\n")
logging.info(f"Total simulation time: {simulation_time}\n")
logging.info(f"Number of nodes: {number_of_nodes}")
logging.info(f"Number of selfish nodes: {number_selfish_nodes}")
logging.info(f"Number of honest nodes: {number_honest_nodes}")

###############################################################################################
######################################### Data import #########################################
###############################################################################################

# IMPORTED OR DATASET GENERATED LAST?
use_import = True

# gets the last created filename (which is the latest dataset)
parent_dir = os.path.dirname(os.getcwd())
search_dir = parent_dir + "/output/data/"
os.chdir(search_dir)
files = filter(os.path.isfile, os.listdir(search_dir))
files = [os.path.join(search_dir, f) for f in files]  # add path to each file
files.sort(key=lambda x: os.path.getmtime(x))
path = files[::-1][0]
fname = os.path.basename(path)

imported_data_filename = "import.csv"
path_import = os.getcwd() + f"/{imported_data_filename}"
