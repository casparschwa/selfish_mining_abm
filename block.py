################
## block data ##
################


class Block:
    def __init__(
        self,
        id,
        miner=None,
        miner_is_selfish=False,
        height=0,
        time=0.0,
        last_update=0.0,
        reached_nodes=0,
        failed_gossip=0,
        main_chain=True,
    ):
        self.id = id  # block id
        self.miner = miner  # miner id
        self.miner_is_selfish = miner_is_selfish  # block mined by selfish/honest node.
        self.height = int(height)  # block height
        self.time = float(time)  # time of block creation
        # time that block was last adopted by another node
        self.last_update = float(last_update)
        # number of nodes that have adopted block
        self.reached_nodes = int(reached_nodes)
        # number of times the block was rejected by peers (also counts if the block was rejected because peer already had received the block before and is part of main chain)
        self.failed_gossip = int(failed_gossip)
        # whether block is part of main chain: True/False (tagging happens after sim.)
        self.main_chain = main_chain

    def asdict(self):
        self.attr_dict = {
            "id": self.id,
            "miner": self.miner,
            "miner_is_selfish": self.miner_is_selfish,
            "height": self.height,
            "time": self.time,
            "last_update": self.last_update,
            "reached_nodes": self.reached_nodes,
            "failed_gossip": self.failed_gossip,
            "main_chain": self.main_chain,
        }
        return self.attr_dict
