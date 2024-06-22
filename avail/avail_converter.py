import numpy as np

from utils.utils import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from observer.observer import Observer


class Avail():
    def __init__(self, observer: "Observer"):
        self.observer = observer

    def pre_avail_actions(self):
        """가공전 avail 정보"""
        
        avail_total_act = np.array(self.observer.env_core.get_avail_actions(), dtype=np.float32)
        assert avail_total_act.shape == (self.observer.n_agents, self.observer.n_actions)
        return avail_total_act

    def avail_act_parsing(self, avail_total_act):
        avail_act = np.zeros((self.observer.n_agents, self.observer.dim_act), dtype=np.float32) # (n_ally, dim_act)
        avail_act[:, NO_OP_IDX] = avail_total_act[:, NO_OP_IDX] # no-op
        avail_act[:, STOP_IDX] = avail_total_act[:, STOP_IDX] # stop
        avail_act[:, MOVE_IDX] = avail_total_act[:, MOVE_IDX: MOVE_IDX+self.observer.dim_move].any(-1) # move
        avail_act[:, TARGET_IDX] = avail_total_act[:, MOVE_IDX+self.observer.dim_move:].any(-1) # target

        return avail_act

    def avail_move_parsing(self, avail_total_act):
        avail_move = np.zeros((self.observer.n_agents, self.observer.dim_move), dtype=np.float32) # (n_ally, dim_move)
        avail_move[:, 0] = avail_total_act[:, MOVE_NORTH_IDX] # north
        avail_move[:, 1] = avail_total_act[:, MOVE_SOUTH_IDX] # south
        avail_move[:, 2] = avail_total_act[:, MOVE_EAST_IDX] # east
        avail_move[:, 3] = avail_total_act[:, MOVE_WEST_IDX] # west

        return avail_move

    def avail_target_parsing(self, avail_total_act):
        avail_target = np.zeros((self.observer.n_agents, self.observer.dim_target), dtype=np.float32) # (n_ally, dim_target)
        avail_target[:, :] = avail_total_act[:, MOVE_IDX+self.observer.dim_move:] # targets

        return avail_target

    def get_avail(self):
        avail_total_act = self.pre_avail_actions()

        avail_dict = {
            "avail_act": self.avail_act_parsing(avail_total_act),
            "avail_move": self.avail_move_parsing(avail_total_act),
            "avail_target": self.avail_target_parsing(avail_total_act),
        }

        return avail_dict
