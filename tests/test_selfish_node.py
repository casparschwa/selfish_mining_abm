import os, sys, inspect
import unittest
import networkx as nx
import numpy as np

# allow import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from blockchain import GillespieBlockchain
from selfish_node import SelfishNode


#################################################################
#################################################################
#################################################################
number_of_nodes = 10
is_selfish = np.append(np.ones(2), np.zeros(8))
hashing_power = np.array([0.2, 0.13, 0.07, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1])
net_p2p = nx.gnm_random_graph(10, 10, seed=2)
nx.draw(net_p2p)
tau_nd = 0.0001  # 0.01 minutes -> 0.6 seconds...
tau_mine = 10
simulating_time = 100
model = GillespieBlockchain(
    net_p2p, is_selfish, hashing_power, tau_nd=tau_nd, tau_mine=tau_mine, verbose=False
)

while model.time < simulating_time:
    model.next_event()

#################################################################
#################################################################
#################################################################
class TestSelfishNode(unittest.TestCase):
    def setUp(self):
        self.n0 = model.nodes[0]
        self.n1 = model.nodes[1]

    def test_set_neighbors(self):
        self.assertEqual(self.n0.neighbors, set([9, 1]))
        self.assertEqual(self.n1.neighbors, set([0, 5]))

    def test_set_selfish_neighbors(self):
        self.assertEqual(self.n0.neighbors, set([9]))

    # def test_set_selfish_neighbors(self):
    #     self.assertEqual(self.n0.selfish_neighbors, set([9]))


if __name__ == "__main__":
    unittest.main()
