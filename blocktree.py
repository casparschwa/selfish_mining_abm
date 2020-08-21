import networkx as nx
import numpy as np
import block


class BlockTree:
    def __init__(self):
        self.tree = nx.DiGraph()  # empty network, gets filled with block creation
        self.tree.add_node(0)  # add one node (genesis block)

        # This will become a dict of dicts {node1: {atrr1: xxx, attr2: yyy, ...}, node2: ...}.
        # Allows to have a memory cheap network, dictionary can always be added as attributes/labels.
        self.attributes = {}

        self.max_height = 0  # height of longest block tree
        # number of blocks created in total (!= height of block tree)
        self.n_blocks = 1

        # create genesis block to initiate block tree
        self.attributes[0] = {
            "id": 0,
            "miner": "genesis",
            "miner_is_selfish": None,
            "height": int(0),
            "time": float(0.0),
            "last_update": float(0.0),
            "reached_nodes": int(0),
            "failed_gossip": int(0),
            "main_chain": True,
        }

    def add_child_block(self, parent_block, miner, miner_is_selfish, time=None):
        """
        Adds a new node to the network: In other words, it adds a block to the parent 
        block, effectively enabling a chain of blocks. 
        """

        # raise error if statement is not true
        assert parent_block < self.n_blocks

        new_block = self.n_blocks
        self.n_blocks += 1

        self.tree.add_node(new_block)
        self.tree.add_edge(parent_block, new_block)

        new_height = self.attributes[parent_block]["height"] + 1

        # call Block class from block.py
        self.attributes[new_block] = block.Block(
            id=new_block,
            miner=miner,
            miner_is_selfish=miner_is_selfish,
            height=new_height,
            time=time,
            last_update=time,
            reached_nodes=1,
            failed_gossip=0,
            main_chain=False,
        ).asdict()

        if new_height > self.max_height:
            self.max_height = new_height

        return new_block

    def tag_main_chain(self):
        """
        This function tags nodes (blocks) that are on the main chain as "main chain". 
        This function is called at the end of the simulation, so it's a one-off process.
        """

        # get last block on longest chain
        main_chain_tail = [
            i
            for i in self.tree.nodes()
            if self.attributes[i]["height"] == self.max_height
        ]

        # IMPORTANT NOTE
        # this just picks the first element of the list, discarding a chain of equal length
        # this is likely no problem, but there could be edge cases

        node_to_tag = main_chain_tail[0]

        while node_to_tag != 0:
            self.attributes[node_to_tag]["main_chain"] = True
            node_to_tag = list(self.tree.predecessors(node_to_tag))[0]

    def nodes(self):
        """
        This function returns a list with the attributes of all nodes of the following form:
        [{dict of n1's attributes}, {dict of n2's attributes}, ...]
        """
        return [self.attributes[n] for n in self.tree.nodes()]

    def subtree_sizes(self):
        """
        This function returns a dictionary with each node's (block's) size of its 
        subtree (count nodes that follow and include yourself). E.g. a 'dead-end' will 
        have a subtree size of 0. Node (block) 0 will have a sub tree size of n_blocks. 
        """

        self.subtree_size_values = {}

        # loop over all blocks created (backwards) and use the respective block as the root for calling __get_subtree_length()
        for n in reversed([n for n in self.tree.nodes()]):
            self.__get_subtree_size(n)

        return np.array(
            [self.subtree_size_values[n] for n in sorted(self.tree.nodes())],
            dtype=np.int32,
        )

    def __get_subtree_size(self, root):
        """
        This private (!) function is the main ingredient for the subtree_sizes() function.
        It looks at the root's successors and increases the size count (subtree_size_count) by its existing 
        value or increases it by 1.
        """

        subtree_size_count = 0
        # note that G.successors() is the same as G.neighbors() for directed graphs!
        for i in self.tree.successors(root):
            # check if block is in dict created in subtree_sizes() from which this fct. is called
            # if block is in dict increase count of subtree_size_count by its value
            if i in self.subtree_size_values:
                subtree_size_count += self.subtree_size_values[i]

            # IMPORTANT NOTE: I think the else-statement is never reached?! Because we're going backwards. So there is no way that a successor of a particular block is not part of the subtree_size_value dict already (when that particular block is used as root for this function)
            else:
                subtree_size_count += self.__get_subtree_size(i)

        self.subtree_size_values[root] = (
            subtree_size_count + 1
        )  # +1 to include the root block.
        return subtree_size_count + 1

    def subtree_lengths(self):
        """
        This function returns an array with the length of the longest chain of successors of 
        each node (block). Essentially, this allows us to get the length of the chain ignoring 
        orphans. It does so mainly by calling __get_subtree_length(root). 
        """
        self.subtree_length_values = {}

        # loop over all blocks created (backwards) and use the respective block as the root for calling __get_subtree_length()
        for n in reversed([n for n in self.tree.nodes()]):
            self.__get_subtree_length(n)

        return np.array(
            [self.subtree_length_values[n] for n in sorted(self.tree.nodes())],
            dtype=np.int32,
        )

    def __get_subtree_length(self, root):
        """
        This private (!) function is the main ingredient for the subtree_lengths() function.
        It looks at the root's successors and counts only the successors that are part of 
        the longest chain, essentially disregarding orphan blocks. 
        """
        max_length = 0
        for i in self.tree.successors(root):
            if i in self.subtree_length_values:
                length = self.subtree_length_values[i]
            else:
                length = self.__get_subtree_length(i)
            if length > max_length:
                max_length = length
        self.subtree_length_values[root] = max_length + 1
        return max_length + 1

    # IMPORTANT NOTE: is_branch() function does not work yet...

    # def is_branch(self):
    #     """
    #     This function returns a vector with booleans that are true if a node is on
    #     the branch (main chain).
    #     """

    #     # initialize vector with zeroes
    #     is_branch = np.zeros(self.tree.number_of_nodes(), dtype=np.bool)

    #     for n in self.tree.nodes():
    #         if self.attributes[n]["main_chain"] == True:
    #             continue
    #         parent = list(self.tree.predecessors(self.attributes[n]["id"]))[0]

    #         # IMPORTANT NOTE: is_main_chain is not defined yet. This happens outside this class!
    #         if is_main_chain[parent] == True:
    #             is_branch[self.attributes[n]["id"]] = True

    #     return is_branch

    def __repr__(self):
        """
        Placeholder function. For now it just prints the attributes. If object was 
        created using test = BlockTree(), then it can be called by "print(test)".
        """
        return "BlockTree({})".format(self.attributes)

    def __getitem__(self, blk):
        """
        This function returns the dict of attributes of the block (blk) passed to function. This allows to access it from e.g. blockchain.py in a much easier way:
        e.g. --> self.nodes[i]["height"] instead of self.nodes.attributes[i]["height]
        """
        assert blk < self.n_blocks
        return self.attributes[blk]

    def __len__(self):
        """
        This function returns the total number of blocks created.
        """
        return self.n_blocks

    def results(self):
        """
        ONLY CALL AT END OF SIMULATION AND AFTER HAVING TAGGED THE MAIN CHAIN
        """

        # tag main chain, otherwise results are not obtainable
        self.tag_main_chain()

        # Booleans for whether block is on main chain.
        is_main_chain = np.array(
            [self.attributes[n]["main_chain"] for n in self.attributes], dtype=np.bool,
        )

        # array of propagation time of blocks
        # (NOTE: orphan blocks have very short prop. times because they tend to reach few nodes)
        propagation_time = np.array(
            [
                self.attributes[n]["last_update"] - self.attributes[n]["time"]
                for n in self.tree.nodes()
            ]
        )
        # array of number of reached nodes for each block
        miners = np.array(
            [self.attributes[n]["reached_nodes"] for n in self.tree.nodes()]
        )
        # array of number of failed gossip for each block
        failed_gossip = np.array(
            [self.attributes[n]["failed_gossip"] for n in self.tree.nodes()]
        )
        # array of booleans whether block was mined by selfish (True) or honest (False)
        selfish = np.array(
            [self.attributes[n]["miner_is_selfish"] for n in self.tree.nodes()]
        )

        reached_nodes = miners.copy()

        # CALCULATE ORPHAN RATE
        # total number of blocks created
        num_blocks = self.tree.number_of_nodes()
        # number of blocks part of main chain
        num_blocks_main_chain = np.count_nonzero(is_main_chain)
        # number of orpahned blocks
        num_blocks_orphaned = num_blocks - num_blocks_main_chain
        # rate of orphaned blocks
        orphaned_block_rate = float(num_blocks_orphaned / num_blocks)
        # number of selfish/honest blocks
        num_blocks_selfish = np.count_nonzero(selfish)
        num_blocks_honest = num_blocks - num_blocks_selfish

        # PROPAGATION TIMES CALUCLATIONS
        # # # # # array of booleans for whether a block has propagated fully or not
        # # # # fully_propagated_blocks = reached_nodes == lcc_size
        # # # # # array of propagation times of blocks that have fully propagated
        # # # # fully_propagated_times = propagation_time[fully_propagated_blocks]

        # # # # # mean/median/min/max time for FULLY PROPAGATED blocks
        # # # # mean_time_fully_propagated = np.mean(fully_propagated_times)
        # # # # median_time_fully_propagated = np.median(fully_propagated_times)
        # # # # min_time_fully_propagated = min(i for i in fully_propagated_times if i>0)
        # # # # max_time_fully_propagated = np.max(fully_propagated_times)

        # mean/median/min/max time of propagation for ALL blocks
        mean_time_propagation = np.mean(propagation_time)
        median_time_propagation = np.median(propagation_time)
        min_time_propagatation = min(i for i in propagation_time if i > 0)
        max_time_propagation = np.max(propagation_time)

        # MINING REWARDS
        selfish_revenue = 0
        honest_revenue = 0

        for block in self.tree.nodes():
            if (
                self.attributes[block]["main_chain"] == True
                and self.attributes[block]["miner_is_selfish"] == True
            ):
                selfish_revenue += 1
            if (
                self.attributes[block]["main_chain"] == True
                and self.attributes[block]["miner_is_selfish"] == False
            ):
                honest_revenue += 1

        relative_selfish_revenue = selfish_revenue / (selfish_revenue + honest_revenue)

        data_point = [
            num_blocks,
            num_blocks_selfish,
            num_blocks_honest,
            num_blocks_main_chain,
            num_blocks_orphaned,
            selfish_revenue,
            honest_revenue,
            relative_selfish_revenue,
            # mean_time_fully_propagated,
            # median_time_fully_propagated,
            # min_time_fully_propagated,
            # max_time_fully_propagated,
            mean_time_propagation,
            median_time_propagation,
            min_time_propagatation,
            max_time_propagation,
        ]

        return data_point
