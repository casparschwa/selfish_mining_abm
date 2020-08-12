class SelfishNode:
    def __init__(self, id, block_tree, eta, current_block=0, verbose=False):

        self.verbose = verbose
        if self.verbose:
            print("SelfishNode() instance created")

        self.id = id
        self.block_tree = block_tree
        self.eta = eta
        self.current_block = current_block
        self.current_height = self.block_tree.attributes[current_block]["height"]

        self.neighbors = set()
        self.non_gossiped_to = set()

        # variables for the selfish mining logic
        self.selfish_neighbors = set()
        self.private_branch_length = 0
        self.public_max_height = self.block_tree.attributes[current_block]["height"]
        self.delta = self.current_height - self.public_max_height

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

    def set_selfish_neighbors(self, selfish_index):
        """
        This function returns a set of selfish neighbors of this SelfishNode instance. 
        """
        self.selfish_neighbors = set(selfish_index)
        self.selfish_neighbors.remove(self.id)

        return self.selfish_neighbors

    def __adopt_received_block(self, emitter, time):
        # adopt received block
        self.current_block = emitter.current_block
        self.current_height = emitter.current_height
        self.block_tree.attributes[self.current_block]["last_update"] = time
        self.block_tree.attributes[self.current_block]["reached_nodes"] += 1

    def __reject_received_block(self, emitter):
        # reject received block
        self.block_tree.attributes[emitter.current_height]["failed_gossip"] += 1

    def __broadcast_to_all(self, except_emitter=None, except_miner=None):
        """
        Description:
        -----------
        This private function updates the set of nodes that need to be broadcasted to, namely ALL NODES. 
        
        Parameters:
        -----------
        except_emitter: int
                        node id of the emitter
        except_miner:   int
                        node id of the miner                 
        """
        # broadcast to ALL NODES
        self.non_gossiped_to = self.neighbors.copy()  # reset gossip list
        # remove emitter from non_gossiped_to set if except_emitter = True
        if except_emitter:
            self.non_gossiped_to.remove(except_emitter)
        # discard miner from non_gossiped_to set if except_miner = True (discarding because miner might be emitter)
        if except_miner:
            self.non_gossiped_to.discard(except_miner)

    def __broadcast_to_selfish(self, except_emitter=None, except_miner=None):
        """
        Description:
        -----------
        This private function updates the set of nodes that need to be broadcasted to, namely only SELFISH NODES. 
        
        Parameters:
        -----------
        except_emitter: int
                        node id of the emitter
        except_miner:   int
                        node id of the miner                 
        """
        # broadcast to SELFISH NODES
        self.non_gossiped_to = self.selfish_neighbors.copy()
        # remove emitter from non_gossiped_to set if except_emitter argument is passed
        if except_emitter:
            self.non_gossiped_to.remove(except_emitter)
        # discard miner from non_gossiped_to set if except_miner argument is passed (discarding because miner might be emitter)
        if except_miner:
            self.non_gossiped_to.discard(except_miner)

    def receive_block(self, emitter, time):

        ## check wether received block was mined by a selfish or honest node

        # The received block was mined by a SELFISH NODE
        if self.block_tree.attributes[emitter.current_block]["miner_is_selfish"]:

            # calculate delta_prev (previous to block receival -- like in the paper)
            self.delta = self.current_height - self.public_max_height

            ## check whether received block's height is greater than my current block's height
            # if received block's height is greater than current block's height, then either SCENARIO (A) or SCENARIO (B)
            if emitter.current_height > self.current_height:

                # increase private branch length
                self.private_branch_length += 1

                ## SCENARIO (B): it was a 1-1 race, selfish nodes find a block -> broadcast to all
                if self.delta == 0 and self.private_branch_length == 2:
                    # adopt received block
                    self.__adopt_received_block(emitter, time)
                    # broadcast to ALL NODES except emitter and miner of received block (if that node happens to be a neighbor)
                    self.__broadcast_to_all(
                        except_emitter=emitter.id,
                        except_miner=self.block_tree.attributes[self.current_block][
                            "miner"
                        ],
                    )
                    # reset private branch length
                    self.private_branch_length = 0
                    if self.verbose:
                        print(
                            "SCENARIO (B): node {} received block {} (height: {}, miner: {}) from node {}".format(
                                self.id,
                                self.current_block,
                                self.current_height,
                                self.block_tree.attributes[self.current_block]["miner"],
                                emitter.id,
                            )
                        )
                    return True

                ## SCENARIO (A): any state but delta=0 and private_branch_length=2
                # elif not (self.delta == 0 and self.private_branch_length == 2):
                else:
                    # adopt received block
                    self.__adopt_received_block(emitter, time)

                    # broadcast only to SELFISH NODES except emitter and miner of received block (nodes should both be selfish miners!)
                    self.__broadcast_to_selfish(
                        except_emitter=emitter.id,
                        except_miner=self.block_tree.attributes[self.current_block][
                            "miner"
                        ],
                    )

                    if self.verbose:
                        print(
                            "SCENARIO (A): node {} received block {} (height: {}, miner: {}) from node {}".format(
                                self.id,
                                self.current_block,
                                self.current_height,
                                self.block_tree.attributes[self.current_block]["miner"],
                                emitter.id,
                            )
                        )
                    # # # assert (
                    # # #     emitter.id in self.selfish_neighbors
                    # # # ), "receiver: {}, emitter: {}, selfish neighbors: {}".format(
                    # # #     self.id, emitter.id, self.selfish_neighbors
                    # # # )
                    # # # assert (
                    # # #     self.block_tree.attributes[self.current_block]["miner"]
                    # # #     in self.selfish_neighbors
                    # # # ), "receiver: {}, miner: {}, selfish neighbors: {}".format(
                    # # #     self.id,
                    # # #     self.block_tree.attributes[self.current_block]["miner"],
                    # # #     self.selfish_neighbors,
                    # # # )

                    return True

            # received block's height is equal to current block's height
            elif emitter.current_height == self.current_height:

                ## if current block was mined by an HONEST NODE, adopt the new block (mined by a SELFISH NODE) and broadcast it to ALL NODES
                # IMPORTANT NOTE: This ensures that a selfish node mines on top of their selfish branch in a 1-1 race.

                # current block was mined by an HONEST NODE
                if (
                    self.block_tree.attributes[self.current_block]["miner_is_selfish"]
                    == False
                ):
                    # adopt received block
                    self.__adopt_received_block(emitter, time)
                    # broadcast to ALL NODES except emitter and miner of received block (if that node happens to be a neighbor)
                    self.__broadcast_to_all(
                        except_emitter=emitter.id,
                        except_miner=self.block_tree.attributes[self.current_block][
                            "miner"
                        ],
                    )
                    # reset private branch length
                    self.private_branch_length = 0
                    if self.verbose:
                        print(
                            "SCENARIO (NA) - replace honest block with selfish block (equal heights): node {} received block {} (height: {}, miner: {}) from node {}".format(
                                self.id,
                                self.current_block,
                                self.current_height,
                                self.block_tree.attributes[self.current_block]["miner"],
                                emitter.id,
                            )
                        )
                    return True

                # if current block was mined by a SELFISH NODE, reject the block -> already mining on selfish branch of same height
                else:
                    self.__reject_received_block(emitter)
                    return False

            # received block's height is less than current block's height
            # elif emitter.current_height < self.current_height:
            else:
                self.block_tree.attributes[emitter.current_height]["failed_gossip"] += 1
                return False

        # The block I am receiving was mined by an HONEST NODE
        else:
            # calculate delta_prev
            self.delta = self.current_height - self.public_max_height

            ## check whether received block's height is greater than current public_max_height
            # received block's height is greater than current public_max_height
            if emitter.current_height > self.public_max_height:
                #
                # IMPORTANT NOTE: implement a test to see if there might be cases where received block's height leads the current public max height by more than 1 block (due to e.g. propagation lag)
                # if this assert error comes up, implement a strategy to prevent this problem
                #
                assert (
                    emitter.current_height - self.public_max_height == 1
                ), "Difference between received block's height and current public max. height is greater than one. It's {}. node id: {}, emitter id: {}".format(
                    emitter.current_height - self.public_max_height, self.id, emitter.id
                )

                # update public max height
                self.public_max_height = emitter.current_height

                ## scenario (c) & (d) & (e):
                if emitter.current_height > self.current_height:
                    # adopt received block
                    self.__adopt_received_block(emitter, time)
                    # broadcast to ALL NODES except emitter and miner of received block (if that node happens to be a neighbor)
                    self.__broadcast_to_all(
                        except_emitter=emitter.id,
                        except_miner=self.block_tree.attributes[self.current_block][
                            "miner"
                        ],
                    )
                    # reset private branch length
                    self.private_branch_length = 0

                    if self.verbose:
                        print(
                            "SCENARIO (C/D/E): node {} received block {} (height: {}, miner: {}) from node {}".format(
                                self.id,
                                self.current_block,
                                self.current_height,
                                self.block_tree.attributes[self.current_block]["miner"],
                                emitter.id,
                            )
                        )

                    return True

                ## SCENARIO (F):
                elif self.delta == 1:
                    # reject received block
                    self.__reject_received_block(emitter)
                    # broadcast current (selfish) block to ALL NODES (including emitter and miner!)
                    self.__broadcast_to_all()
                    # reset private branch length
                    self.private_branch_length = 0

                    if self.verbose:
                        print(
                            "SCENARIO (F): node {} rejected received block {} (height: {}, miner: {}) from node {} AND published block {} to ALL nodes".format(
                                self.id,
                                emitter.current_block,
                                emitter.current_height,
                                self.block_tree.attributes[self.current_block]["miner"],
                                emitter.id,
                                self.current_block,
                            )
                        )

                    return False

                ## SCENARIO (G):
                elif self.delta == 2:
                    # reject received block
                    self.__reject_received_block(emitter)
                    # broadcast current (selfish) block to ALL NODES (including emitter and miner!)
                    self.__broadcast_to_all()
                    # reset private branch length
                    self.private_branch_length = 0

                    if self.verbose:
                        print(
                            "SCENARIO (G): node {} rejected received block {} (height: {}, miner: {}) from node {} AND published block {} to ALL nodes".format(
                                self.id,
                                emitter.current_block,
                                emitter.current_height,
                                self.block_tree.attributes[self.current_block]["miner"],
                                emitter.id,
                                self.current_block,
                            )
                        )

                    return False

                ## SCENARIO (H):
                else:
                    if self.verbose:
                        print("SCENARIO H")
                    return False

    def mine_block(self, time=None):
        """
        This function adds a block to the block tree and updates the gossiping list
        and height respectively. 
        """
        # update current block (add_child_block() returns a new block)
        self.current_block = self.block_tree.add_child_block(
            parent_block=self.current_block,
            miner=self.id,
            miner_is_selfish=True,
            time=time,
        )

        #%%%%
        # calculate delta_prev (previous to block receival -- like in the paper)
        self.delta = self.current_height - self.public_max_height
        # update current height
        self.current_height = self.block_tree[self.current_block]["height"]
        # increase private branch length
        self.private_branch_length += 1

        ## SCENARIO (B): it was a 1-1 race, selfish nodes find a block -> broadcast to all
        if self.delta == 0 and self.private_branch_length == 2:
            # broadcast to ALL NODES except emitter and miner of received block (if that node happens to be a neighbor)
            self.__broadcast_to_all()
            # reset private branch length
            self.private_branch_length = 0

        ## SCENARIO (A): any state but delta=0 and private_branch_length=2
        # elif not (self.delta == 0 and self.private_branch_length == 2):
        else:
            # broadcast only to SELFISH NODES except emitter and miner of received block (nodes should both be selfish miners!)
            self.__broadcast_to_selfish()
        #%%%%

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

#