class HonestNode:
    def __init__(self, id, block_tree, eta, current_block=0, verbose=False):

        if verbose:
            print("HonestNode() instance created")

        self.id = id
        self.block_tree = block_tree
        self.eta = eta
        self.current_block = current_block
        self.verbose = verbose

        self.neighbors = set()
        self.non_gossiped_to = set()

        self.current_height = self.block_tree.attributes[current_block]["height"]

    def set_neighbors(self, ls_n):
        """
        Function returns a set of neighbors of specified node. The argument is 
        self.net_p2p.neighbors(n). It works like this:
        ---
        It is called in the GillespieBlockchain() class like this: 
        ---
            for n in self.net_p2p.nodes():
                self.nodes[n].set_neighbours(self.net_p2p.neighbors(n))
        ---
        So the argument of set_neighbours() is self.net_p2p.neighbors(n), where net_p2p
        is something like the following (it creates a graph with a certain degree distr.)
        ---
            net_p2p = nx.random_degree_sequence_graph([number_of_neighbors for i in range(number_of_nodes)])
        ---
        Hence, net_p2p.neighbors(n) returns an iterator over the neighbors of node n in this
        generated network (net_p2p). 
        """
        self.neighbors = set(ls_n)
        return self.neighbors

    def receive_block(self, emitter, time):
        """
        This function handles the receival of a block: It checks whether "my" current 
        height is greater than that of the block received. If it is, it accepts it, 
        otherwise the block is rejected.
        """
        # check whether emitter of block is selfish or honest so that we know whether to use emitter.current_block (honest nodes) or emitter.block_to_broadcast_to_honest (selfish nodes)
        # emitter is honest
        if isinstance(emitter, HonestNode):
            block = emitter.current_block
            height = emitter.current_height
        # emitter is selfish
        else:
            block = emitter.block_to_broadcast_to_honest
            height = emitter.height_to_broadcast_to_honest

        # if "my" current height is larger than the emitter's, reject block (increase "failed_gossip" counter of the emitter's block)
        if self.current_height >= height:
            self.block_tree[block]["failed_gossip"] += 1
            if self.verbose:
                print(
                    "---- Gossip failed: node {} rejected block {} (height: {}, miner: {}) from node {}, and continues mining on {} (height: {})".format(
                        self.id,
                        block,
                        height,
                        emitter.block_tree[block]["miner"],
                        emitter.id,
                        self.current_block,
                        self.current_height,
                    )
                )
            return False

        # if received block's height is greater than "mine", then adopt it:
        # Update to new block height and recipient adds all neighbors (except emitter)
        # to non_gossiped_to set
        else:
            self.current_block = block
            self.current_height = height
            self.block_tree.attributes[self.current_block]["last_update"] = time
            self.block_tree.attributes[self.current_block]["reached_nodes"] += 1
            self.non_gossiped_to = self.neighbors.copy()  # reset gossip list
            self.non_gossiped_to.remove(emitter.id)

            if self.verbose:
                print(
                    "Node {} adopts block {} (height: {}, miner: {}) from node {}".format(
                        self.id,
                        self.current_block,
                        self.current_height,
                        self.block_tree[self.current_block]["miner"],
                        emitter.id,
                    )
                )
            return True

    def mine_block(self, time=None):
        """
        This function adds a block to the block tree and updates the gossiping list
        and height respectively. 
        """
        # update current block (add_child_block() returns a new block)
        self.current_block = self.block_tree.add_child_block(
            parent_block=self.current_block,
            miner=self.id,
            miner_is_selfish=False,
            time=time,
        )
        self.current_height = self.block_tree[self.current_block]["height"]
        self.non_gossiped_to = self.neighbors.copy()  # reset gossip list
        return self.current_block

    def gossiped_to(self, n_i):
        """
        This function removes a node from the gossiping list 
        """
        self.non_gossiped_to.remove(n_i)

    def is_gossiping(self):
        """
        This function retuns True if node still has to gossip to some neighbors, i.e. 
        the node is still gossiping and it returns False if it is not gossiping anymore. 
        """
        return len(self.non_gossiped_to) > 0
