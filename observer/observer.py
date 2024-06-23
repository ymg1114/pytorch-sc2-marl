import numpy as np

from utils.utils import *
from avail.avail_converter import Avail

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smacv2.env.starcraft2.starcraft2 import StarCraft2Env


class Observer():
    def __init__(self, parent: "StarCraft2Env"):
        self.env_core = parent
        self.env_info = self.env_core.get_env_info()
        self.n_actions = self.env_info["n_actions"] # 총 행동 개수
        self.n_agents = self.env_info["n_agents"] # 총 아군 초기 인원수
        self.n_enemies = self.env_core.n_enemies # 총 적군 초기 인원수

        # TODO: 하드코드 디멘션 구성. SMAC2 원격 기본 환경의 변화에 취약
        self.dim_act = len(ACT) # no-op, stop, move, target
        self.dim_move = len(MOVE) # north, south, east, west
        self.dim_target = self.n_actions - 2 - len(MOVE) # max(n_ally, n_enemy)
        assert self.dim_target == max(self.n_agents, self.n_enemies)

        self.obs_size = self.env_core.get_obs_size()
        self.obs_feature_names = self.env_core.get_obs_feature_names()

        self.move_feats_dim = self.env_core.get_obs_move_feats_size()
        self.own_feats_dim = self.env_core.get_obs_own_feats_size()
        self.ally_feats_dim = self.env_core.get_obs_ally_feats_size()
        self.enemy_feats_dim = self.env_core.get_obs_enemy_feats_size()

        self.avail = Avail(self)

    def get_obs(self):
        """가공전 obs 정보를 활용"""
        
        obs_total = np.array(self.env_core.get_obs(), dtype=np.float32)
        assert obs_total.shape == (self.n_agents, self.obs_size)

        n_allies, n_ally_feats = self.ally_feats_dim
        n_enemies, n_enemy_feats = self.enemy_feats_dim

        obs_mine = self.get_obs_mine(obs_total)
        obs_ally = self.get_obs_ally(obs_total)
        obs_enemy = self.get_obs_enemy(obs_total)

        assert obs_mine.shape == (self.n_agents, self.move_feats_dim+self.own_feats_dim)
        assert obs_ally.shape == (self.n_agents, n_allies*n_ally_feats)
        assert obs_enemy.shape == (self.n_agents, n_enemies*n_enemy_feats)

        obs_dict = {
            "obs_mine": obs_mine,
            "obs_ally": obs_ally,
            "obs_enemy": obs_enemy,
        }
        return obs_dict

    def get_obs_mine(self, obs_total):
        """주의) 하드 코드 성향이 존재
        move-feat -> enemy-feat -> ally-feat -> own-feat
        순으로 obs_total (2d-ndarray) 가 구성됨. 이를 슬라이싱
        """
        
        mine_move_feat = obs_total[:, :self.move_feats_dim]
        mine_own_feat = obs_total[:, -self.own_feats_dim:]
        return np.hstack((mine_move_feat, mine_own_feat))

    def get_obs_enemy(self, obs_total):
        """주의) 하드 코드 성향이 존재
        move-feat -> enemy-feat -> ally-feat -> own-feat
        순으로 obs_total (2d-ndarray) 가 구성됨. 이를 슬라이싱
        """
        
        n_enemies, n_enemy_feats = self.enemy_feats_dim
        src = self.move_feats_dim
        dst = self.move_feats_dim + n_enemies*n_enemy_feats
        return obs_total[:, src: dst]

    def get_obs_ally(self, obs_total):
        """주의) 하드 코드 성향이 존재
        move-feat -> enemy-feat -> ally-feat -> own-feat
        순으로 obs_total (2d-ndarray) 가 구성됨. 이를 슬라이싱
        """
        
        n_allies, n_ally_feats = self.ally_feats_dim
        src = -self.own_feats_dim - n_allies*n_ally_feats
        dst = -self.own_feats_dim
        return obs_total[:, src: dst]

    def get(self):
        out_dict = {
            **self.avail.get_avail(),
            **self.get_obs(),
        }
        return out_dict
