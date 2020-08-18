import networkx as nx
import numpy as np
import blocktree
import block
import honest_node
import selfish_node


class GillespieBlockchain:
    #%%%%
    # one more variable should be passed to __init__(), e.g. is_selfish: add a vector with booleans. This indicated whether a node is honest or selfish.
    #%%%%
    def __init__(
        self, net_p2p, is_selfish, hashing_power, tau_nd, tau_mine=1.0, verbose=False
    ):
        self.verbose = verbose
        if verbose:
            print("GillespieBlockchain() instance created")

        # make sure input data matches
        assert net_p2p.number_of_nodes() == len(hashing_power) == len(is_selfish)
        assert sum(hashing_power) == 1

        self.block_tree = blocktree.BlockTree()  # initialize block tree

        #%%%%
        # array of booleans: True -> node is selfish / False -> node is honest
        self.is_selfish = is_selfish

        # create index lists with the indexes of all selfish nodes and honest nodes respectively
        self.selfish_index = [
            index for index, value in enumerate(self.is_selfish) if value == True
        ]
        self.honest_index = [
            index for index, value in enumerate(self.is_selfish) if value == False
        ]

        # create list of HonestNode and SelfishNode objects respectively
        self.nodes = []
        for index, selfish in enumerate(self.is_selfish):
            if selfish == True:
                self.nodes.append(
                    selfish_node.SelfishNode(
                        index,
                        self.block_tree,
                        eta=hashing_power[index],
                        current_block=0,
                        verbose=self.verbose,
                    )
                )
            else:
                self.nodes.append(
                    honest_node.HonestNode(
                        index,
                        self.block_tree,
                        eta=hashing_power[index],
                        verbose=self.verbose,
                    )
                )

        self.net_p2p = net_p2p  # network with certain degree distribution
        # NOTE: Assumption is that all selfish miners selfish miners are connected so we manually add edges between all selfish nodes.
        edge_tuples_list = [
            (i, j) for i in self.selfish_index for j in self.selfish_index if i != j
        ]  # list of all possible tuples of selfish nodes
        self.net_p2p.add_edges_from(
            edge_tuples_list
        )  # adds edges between all selfish nodes
        #%%%%

        self.gossiping_nodes = set()  # initialize an empty set for gossiping nodes

        # creates set of neighbors for all the Node() instances in self.nodes
        for n in self.net_p2p.nodes():
            self.nodes[n].set_neighbors(self.net_p2p.neighbors(n))

        # create set of selfish neighbors for all SelfishNode instances
        for n in self.selfish_index:
            self.nodes[n].set_selfish_neighbors(self.selfish_index)

        # returns an array with indexes of nodes for which hashing_power is >0.
        (self.hashing_nodes,) = np.where(hashing_power > 0)
        # Set up hashing power vectors and normalize them
        self.hashing_power = hashing_power[self.hashing_nodes]
        self.hashing_power /= np.sum(self.hashing_power)

        # Parameters
        # avg. time of network diffusion (network delay / latency)
        self.tau_nd = tau_nd
        self.tau_mine = tau_mine  # avg. time between blocks -> bitcoin: 10'
        self.lambda_nd = 1.0 / tau_nd  # lambda: network delay (latency)
        self.lambda_mine = 1.0 / tau_mine  # lambda: mining

        self.time = 0.0
        self.time_in_consensus = 0.0

    def __mine_event(self):
        """
        This function handles a mining event. 
        """
        # pick miner of new block and call mine_block() function
        miner = np.random.choice(self.hashing_nodes, p=self.hashing_power)
        self.nodes[miner].mine_block(self.time)

        # the if-statement is actually unnecessary, because mine_block() also resets the non_gossiped_to set. Therefore is_gossiping() will always return True.
        # add miner to set of gossiping nodes
        if self.nodes[miner].is_gossiping() == True:
            self.gossiping_nodes.add(miner)

    def __gossip_event(self):
        """
        This function handles a gossip event: It randomly picks an emitter and recipient from
        the gossiping_nodes set and set of non_gossiped_to nodes (of emitter) respectively.
        It then handles the actions of the emitter and recipients, i.e. adoption/rejection
        of block by receiver and updates the gossiping sets etc. 
        """
        # pick random emitter from set of gossiping nodes
        emitter = np.random.choice(list(self.gossiping_nodes))
        # pick random recipient from set of nodes that have not been gossiped to
        recipient = np.random.choice(list(self.nodes[emitter].non_gossiped_to))

        # # # assert (
        # # #     emitter == recipient
        # # # ), "emitter id: {}; emitter non_gossiped_to: {}".format(
        # # #     self.nodes[emitter].id, self.nodes[emitter].non_gossiped_to
        # # # )

        # removes the recipient from the non_gossiped set for the emitter
        self.nodes[emitter].gossiped_to(recipient)

        # recipient receives block from emitter (HonestNode and SelfishNode have different receive_block functions)
        self.nodes[recipient].receive_block(self.nodes[emitter], self.time)

        # If the emitter is not gossiping anymore (after having gossiped to current recipient), then remove emitter from set of gossiping nodes.
        if not self.nodes[emitter].is_gossiping():
            self.gossiping_nodes.remove(emitter)
            # if emitter was selfish and informing, stop informing
            if isinstance(self.nodes[emitter], selfish_node.SelfishNode):
                self.nodes[emitter].informing = False

        # If recipient is gossiping, add recipient to gossiping node set
        if self.nodes[recipient].is_gossiping():
            self.gossiping_nodes.add(recipient)

        # If recipient is not gossiping (if-statement above), and if recipient is in gossiping_nodes set,
        # then remove emitter from gossiping nodes set
        elif recipient in self.gossiping_nodes:
            self.gossiping_nodes.remove(recipient)

    def snapshot(self):

        string = "-------------------------\nSNAPSHOT \n"
        for node in self.nodes:
            string += "node {} ({}) is mining on block {} (height {}) \n".format(
                node.id,
                "selfish" if self.is_selfish[node.id] else "honest",
                node.current_block,
                node.current_height,
            )
        string += "-------------------------"
        print(string)

    def next_event(self):
        """
        This function ...
        """

        #
        # IMPORTANT NOTE: Why is only lambda_gossip summed over the number of nodes? Because gossiping is not dependent on a global average like mining is with 10 minute intervals?
        #

        # computes waiting time before next event occurs
        lambda_gossip = len(self.gossiping_nodes) * self.lambda_nd
        lambda_sum = self.lambda_mine + lambda_gossip

        # generate waiting time for next event
        time_increment = -np.log(np.random.random()) / lambda_sum

        # count the number of tails (number of chain splits at equal height)
        tail_blocks = set([node.current_block for node in self.nodes])
        # if everyone is on the same block, there is consensus -> return True
        is_consensus = len(tail_blocks) == 1
        if is_consensus:
            self.time_in_consensus += time_increment

        # jump to event time generated above.
        self.time += time_increment

        #
        # IMPORTANT NOTE: Should we not use the same np.random.random() value from time_increment above for the state update? Should we not have only one source of randomness?!
        # What is the exact logic of lambda_mine / lambda_sum below?
        #

        # update the state (either mining or gossiping event)
        rnd_event = np.random.random()

        if rnd_event <= (self.lambda_mine / lambda_sum):
            # if event is mining
            self.__mine_event()

            # print snapshots
            if self.verbose:
                self.snapshot()
        else:
            # if event is gossip
            self.__gossip_event()
