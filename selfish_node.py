import logging


class SelfishNode:
    def __init__(self, id, block_tree, eta, current_block=0, verbose=False):

        self.verbose = verbose
        if self.verbose:
            logging.info("SelfishNode() instance created")

        self.id = id
        self.block_tree = block_tree
        self.eta = eta  # hashing power
        self.current_block = current_block
        self.current_height = self.block_tree[current_block]["height"]

        self.neighbors = set()
        self.non_gossiped_to = set()

        ## variables required for SELFISH MINING LOGIC

        # set of selfish nodes (neighbors)
        self.selfish_neighbors = set()

        # variables to determine which scenario is taking place
        self.private_branch_length = 0
        self.public_max_height = self.block_tree[current_block]["height"]
        self.delta = self.current_height - self.public_max_height

        # selfish nodes may broadcast an old block to honest nodes that is different to the current block -> scenario (H)
        self.block_to_broadcast_to_honest = self.current_block
        self.height_to_broadcast_to_honest = self.current_height

        # in scenario (f), (g) and (h) a selfish node must inform its selfish peers about the receival of an honest block, by default this variable is False.
        self.informing = False

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
        self.block_tree[self.current_block]["last_update"] = time
        self.block_tree[self.current_block]["reached_nodes"] += 1

        # # # ## reset block_to_broadcast_to_honest to ensure node always sends correct block to HONEST NODES
        # # # self.block_to_broadcast_to_honest = self.current_block
        # # # self.height_to_broadcast_to_honest = self.current_height

    def __adopt_received_information(self, emitter):
        # adopt information
        # importantly we must update the public max height as well!
        self.public_max_height = emitter.public_max_height
        self.block_to_broadcast_to_honest = emitter.block_to_broadcast_to_honest
        self.height_to_broadcast_to_honest = emitter.height_to_broadcast_to_honest

    def __reject_received_block(self, emitter):
        # reject received block
        self.block_tree[emitter.current_height]["failed_gossip"] += 1

    def __override_received_block(self):
        """
        ...
        """
        ## broadcast old selfish block that matches new max_public_height of honest block (that I just received)
        # to ensure that the old block we want to broadcast is actually a parent of the selfish node's current block nx.predecessors() fct. is used (similar to tag_main_chain() in blocktree.py)
        # find parent block that matches current public max. height and update block_to_broadcast_to_honest accordingly (analogously for height...)

        # start going back from current block
        block = self.current_block
        height = self.current_height

        # iterate until you find parent block with height equal to public max height
        while height != self.public_max_height:
            block = list(self.block_tree.tree.predecessors(block))[0]
            height = self.block_tree[block]["height"]

        # update block/height to broadcast to honest
        self.block_to_broadcast_to_honest = block
        self.height_to_broadcast_to_honest = height

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
        if except_emitter is not None:
            self.non_gossiped_to.discard(except_emitter)
        # discard miner from non_gossiped_to set if except_miner = True (discarding because miner might be emitter)
        if except_miner is not None:
            self.non_gossiped_to.discard(except_miner)

        # reset private branch length
        self.private_branch_length = 0

    def __broadcast_to_selfish(self, except_emitter=None, except_miner=None):
        """
        Description:
        -----------
        This private function updates the set of nodes that need to be broadcasted to, namely only SELFISH NODES. 
        
        Parameters:
        -----------
        except_emitter: int (default=None)
                        node id of the emitter
        except_miner:   int (default=None)
                        node id of the miner                 
        """
        # broadcast to SELFISH NODES
        self.non_gossiped_to = self.selfish_neighbors.copy()
        # remove emitter from non_gossiped_to set if except_emitter argument is passed
        if except_emitter is not None:
            self.non_gossiped_to.discard(except_emitter)
        # discard miner from non_gossiped_to set if except_miner argument is passed (discarding because miner might be emitter)
        if except_miner is not None:
            self.non_gossiped_to.discard(except_miner)

    def __broadcast_to_honest(self, except_emitter=None, except_miner=None):

        # broadcast to HONEST NODES
        self.non_gossiped_to = self.neighbors.copy() - self.selfish_neighbors.copy()
        # remove emitter from non_gossiped_to set if except_emitter argument is passed
        if except_emitter is not None:
            self.non_gossiped_to.discard(except_emitter)
        # discard miner from non_gossiped_to set if except_miner argument is passed (discarding because miner might be emitter)
        if except_miner is not None:
            self.non_gossiped_to.discard(except_miner)

    def __broadcast_to_honest_inform_selfish(
        self, except_emitter=None, except_miner=None
    ):
        # broadcast to HONEST NODES & inform SELFISH NODES (ALL NODES receive either block (honest) or information about the block (honest))
        self.non_gossiped_to = self.neighbors.copy()
        # remove emitter from non_gossiped_to set if except_emitter argument is passed
        if except_emitter is not None:
            self.non_gossiped_to.discard(except_emitter)
        # discard miner from non_gossiped_to set if except_miner argument is passed (discarding because miner might be emitter)
        if except_miner is not None:
            self.non_gossiped_to.discard(except_miner)

    def receive_block(self, emitter, time):

        # if emitter is selfish & informing start broadcasting/informating (only if you haven not been informed yet -> second condition)
        if isinstance(emitter, SelfishNode) and emitter.informing == True:
            # I have not heard about this block before
            if emitter.public_max_height > self.public_max_height:
                # adopt information
                self.__adopt_received_information(emitter)
                # start informing your peers
                self.informing = True
                self.__broadcast_to_honest_inform_selfish(except_emitter=emitter.id)
                if self.verbose:
                    logging.info(
                        "node {} received information from node {}".format(
                            self.id, emitter.id
                        )
                    )
                return
            # I know about this block already
            else:
                if self.verbose:
                    logging.info(
                        "node {} received information from node {}, but is already informing its selfish peers".format(
                            self.id, emitter.id
                        )
                    )
                return

        ## check wether received block was mined by a selfish or honest node

        # The received block was mined by a SELFISH NODE
        if self.block_tree[emitter.current_block]["miner_is_selfish"]:

            # calculate delta_prev (previous to block receival -- like in the paper)
            self.delta = self.current_height - self.public_max_height

            ## check whether received block's height is greater than my current block's height
            # if received block's height is greater than current block's height, then either SCENARIO (A) or SCENARIO (B)
            if emitter.current_height > self.current_height:

                # increase private branch length
                self.private_branch_length += 1

                ########################################################################################################################################
                ## SCENARIO (B) ## - it was a 1-1 race, selfish nodes find a block -> broadcast to all
                ########################################################################################################################################

                if self.delta == 0 and self.private_branch_length == 2:
                    # adopt received block
                    self.__adopt_received_block(emitter, time)
                    # broadcast to ALL NODES except emitter and miner of received block (if that node happens to be a neighbor)
                    self.__broadcast_to_all(
                        except_emitter=emitter.id,
                        except_miner=self.block_tree[self.current_block]["miner"],
                    )

                    # update block_to_broadcast_to_honest
                    self.block_to_broadcast_to_honest = self.current_block
                    self.height_to_broadcast_to_honest = self.current_height

                    if self.verbose:
                        logging.info(
                            "SCENARIO (B): node {} adopted block {} from node {} (height: {}, miner: {})".format(
                                self.id,
                                self.current_block,
                                emitter.id,
                                self.current_height,
                                self.block_tree[self.current_block]["miner"],
                            )
                        )
                    return True

                ########################################################################################################################################
                ## SCENARIO (A) ## - any state but delta=0 and private_branch_length=2
                ########################################################################################################################################

                # elif not (self.delta == 0 and self.private_branch_length == 2):
                else:
                    # adopt received block
                    self.__adopt_received_block(emitter, time)

                    # broadcast only to SELFISH NODES except emitter and miner of received block (nodes should both be selfish miners!)
                    self.__broadcast_to_selfish(
                        except_emitter=emitter.id,
                        except_miner=self.block_tree[self.current_block]["miner"],
                    )

                    if self.verbose:
                        logging.info(
                            "SCENARIO (A): node {} adopted block {} from node {} (height: {}, miner: {})".format(
                                self.id,
                                self.current_block,
                                emitter.id,
                                self.current_height,
                                self.block_tree[self.current_block]["miner"],
                            )
                        )

                    return True

            # received block's height is equal to current block's height
            elif emitter.current_height == self.current_height:

                ## if current block was mined by an HONEST NODE, adopt the new block (mined by a SELFISH NODE) and broadcast it to ALL NODES
                # NOTE: This ensures that a selfish node mines on top of their selfish branch in a 1-1 race.

                # current block was mined by an HONEST NODE
                if self.block_tree[self.current_block]["miner_is_selfish"] == False:
                    # adopt received block
                    self.__adopt_received_block(emitter, time)
                    # broadcast to ALL NODES except emitter and miner of received block (if that node happens to be a neighbor)
                    self.__broadcast_to_all(
                        except_emitter=emitter.id,
                        except_miner=self.block_tree[self.current_block]["miner"],
                    )

                    # update block_to_broadcast_to_honest
                    self.block_to_broadcast_to_honest = self.current_block
                    self.height_to_broadcast_to_honest = self.current_height

                    if self.verbose:
                        logging.info(
                            "SCENARIO (N/A): node {} adopted block {} from node {} (height: {}, miner: {}), throwing away honest block of equal height".format(
                                self.id,
                                self.current_block,
                                emitter.id,
                                self.current_height,
                                self.block_tree[self.current_block]["miner"],
                            )
                        )
                    return True

                # if current block was mined by a SELFISH NODE, reject the block -> already mining on selfish branch of same height
                else:
                    self.__reject_received_block(emitter)

                    if self.verbose:
                        logging.info(
                            "SCENARIO (N/A): node {} rejected block {} from node {} (height: {}, miner: {})".format(
                                self.id,
                                emitter.current_block,
                                emitter.id,
                                emitter.current_height,
                                emitter.block_tree[emitter.current_block]["miner"],
                            )
                        )

                    return False

            # received block's height is less than current block's height
            # elif emitter.current_height < self.current_height:
            else:
                self.block_tree[emitter.current_height]["failed_gossip"] += 1

                if self.verbose:
                    logging.info(
                        "SCENARIO (N/A): node {} rejected block {} from node {} (height: {}, miner: {})".format(
                            self.id,
                            emitter.current_block,
                            emitter.id,
                            emitter.current_height,
                            emitter.block_tree[emitter.current_block]["miner"],
                        )
                    )
                return False

        # The block I am receiving was mined by an HONEST NODE
        else:
            # calculate delta_prev
            self.delta = self.current_height - self.public_max_height

            ## check whether received block's height is greater than current public_max_height
            # received block's height is greater than current public_max_height
            if emitter.current_height > self.public_max_height:

                # update public max height
                self.public_max_height = emitter.current_height

                ########################################################################################################################################
                ## scenario (C),(D) and (E) ## - honest nodes are leading, adopt their branch and mine on new head
                ########################################################################################################################################
                if emitter.current_height > self.current_height:
                    # adopt received block
                    self.__adopt_received_block(emitter, time)
                    # broadcast to ALL NODES except emitter and miner of received block (if that node happens to be a neighbor)
                    self.__broadcast_to_all(
                        except_emitter=emitter.id,
                        except_miner=self.block_tree[self.current_block]["miner"],
                    )

                    # update block_to_broadcast_to_honest
                    self.block_to_broadcast_to_honest = self.current_block
                    self.height_to_broadcast_to_honest = self.current_height

                    if self.verbose:
                        logging.info(
                            "SCENARIO (C/D/E): node {} adopted block {} from node {} (height: {}, miner: {})".format(
                                self.id,
                                self.current_block,
                                emitter.id,
                                self.current_height,
                                self.block_tree[self.current_block]["miner"],
                            )
                        )

                    return True
                ########################################################################################################################################
                ## SCENARIO (F) ## - lead was 1, now it's 1-1, publish branch, try our luck:
                ########################################################################################################################################
                elif self.delta == 1:
                    # reject received block
                    self.__reject_received_block(emitter)

                    # broadcast current (selfish) block to HONEST NODES (including emitter and miner!)
                    self.__broadcast_to_honest()

                    # update block_to_broadcast_to_honest
                    self.block_to_broadcast_to_honest = self.current_block
                    self.height_to_broadcast_to_honest = self.current_height

                    if self.verbose:
                        logging.info(
                            "SCENARIO (F): node {} rejected block {} from node {} (height: {}, miner: {}) & published block {} (height: {}) to ALL nodes".format(
                                self.id,
                                emitter.current_block,
                                emitter.id,
                                emitter.current_height,
                                emitter.block_tree[emitter.current_block]["miner"],
                                self.current_block,
                                self.current_height,
                            )
                        )

                    return False
                ########################################################################################################################################
                ## SCENARIO (G) ## - lead was 2, now honest are catching up, publish entire branch:
                ########################################################################################################################################
                elif self.delta == 2:
                    # reject received block
                    self.__reject_received_block(emitter)

                    # broadcast current (selfish) block to HONEST NODES (including emitter and miner!)
                    self.__broadcast_to_honest()

                    # update block_to_broadcast_to_honest
                    self.block_to_broadcast_to_honest = self.current_block
                    self.height_to_broadcast_to_honest = self.current_height

                    if self.verbose:
                        logging.info(
                            "SCENARIO (G): node {} rejected block {} from node {} (height: {}, miner: {}) & published block {} (height: {}) to HONEST nodes".format(
                                self.id,
                                emitter.current_block,
                                emitter.id,
                                emitter.current_height,
                                emitter.block_tree[emitter.current_block]["miner"],
                                self.current_block,
                                self.current_height,
                            )
                        )

                    return False

                ########################################################################################################################################
                ## SCENARIO (H) ## - lead was more than 2, now honest found one block, publish old block that matches public max height:
                ########################################################################################################################################
                # else:
                elif self.delta > 2:
                    # reject received block
                    self.__reject_received_block(emitter)

                    # find old selfish parent block that matches public max height
                    self.__override_received_block()

                    # start informing & broadcast old selfish parent block to HONEST NODES & inform SELFISH NODES
                    self.informing = True
                    self.__broadcast_to_honest_inform_selfish(except_emitter=emitter.id)

                    if self.verbose:
                        logging.info(
                            "SCENARIO (H): node {} rejected block {} from node {} (height: {}, miner: {}) & published block {} (height: {}) to HONEST nodes".format(
                                self.id,
                                emitter.current_block,
                                emitter.id,
                                emitter.current_height,
                                emitter.block_tree[emitter.current_block]["miner"],
                                self.block_to_broadcast_to_honest,
                                self.height_to_broadcast_to_honest,
                            )
                        )
                    return False

            # received block's height is smaller or equal to my current public_max_height -> I have already heard of this block before
            # emitter.current_height <= self.public_max_height:
            else:
                # reject received block
                self.__reject_received_block(emitter)

                if self.verbose:
                    logging.info(
                        "SCENARIO (NA): node {} rejected block {} from node {} (height: {}, miner: {}), because it knew about it already.".format(
                            self.id,
                            emitter.current_block,
                            emitter.id,
                            emitter.current_height,
                            emitter.block_tree[emitter.current_block]["miner"],
                        )
                    )
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
            # stop informing
            # broadcast to ALL NODES (neighbors)
            self.__broadcast_to_all()
            # reset private branch length
            self.private_branch_length = 0

        ## SCENARIO (A): any state but delta=0 and private_branch_length=2
        # elif not (self.delta == 0 and self.private_branch_length == 2):
        else:
            # broadcast only to SELFISH NODES
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
