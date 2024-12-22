import jax.numpy as jnp

from utils.utils import *
from .avail.avail_converter import Avail

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smacv2.env.starcraft2.starcraft2 import StarCraft2Env

from .observer_lib import _jit_update_hp_vmap, _jit_get_obs_vmap


class Observer():
    def __init__(self, parent: "StarCraft2Env"):
        self.env_core = parent
        self.env_info = self.env_core.get_env_info()
        
        self.n_actions = self.env_info["n_actions"] # 총 행동 개수
        self.n_agents = self.env_info["n_agents"] # 총 아군 초기 인원수
        self.n_enemies = self.env_core.n_enemies # 총 적군 초기 인원수

        # TODO: 하드코드 디멘션 구성. SMAC2 원격 기본 환경의 변화에 취약
        self.dim_act = len(ACT) # no-op, stop, move, target, flee
        self.dim_move = len(MOVE) # north, south, east, west
        self.dim_target = self.n_actions - 3 - len(MOVE) # max(n_ally, n_enemy)
        assert self.dim_target == max(self.n_agents, self.n_enemies)

        self.obs_size = self.env_core.get_obs_size()
        self.obs_feature_names = self.env_core.get_obs_feature_names()

        self.move_feats_dim = self.env_core.get_obs_move_feats_size()
        self.own_feats_dim = self.env_core.get_obs_own_feats_size()
        
        self.ally_feats_dim = self.env_core.get_obs_ally_feats_size()
        self.n_allies, self.n_ally_feats = self.ally_feats_dim
        
        self.enemy_feats_dim = self.env_core.get_obs_enemy_feats_size()
        n_enemies, self.n_enemy_feats = self.enemy_feats_dim
        assert self.n_enemies == n_enemies
        assert self.n_agents == self.n_allies + 1
        self.num_units = 1 + self.n_allies + self.n_enemies  # 나, 아군, 적군
        
        self.mine_feats_names = self.obs_feature_names[:self.move_feats_dim] + self.obs_feature_names[-self.own_feats_dim:]
        self.mine_feats_names += ["mean_own_health_delta"]
        
        self.ally_feats_names = self.obs_feature_names[-self.own_feats_dim - self.n_allies*self.n_ally_feats: -self.own_feats_dim]
        self.ally_feats_names += [f"mean_ally_health_delta_{i+1}" for i in range(self.n_allies)]

        self.enemy_feats_names = self.obs_feature_names[self.move_feats_dim: self.move_feats_dim + self.n_enemies*self.n_enemy_feats]
        self.enemy_feats_names += [f"mean_enemy_health_delta_{i}" for i in range(self.n_enemies)]
        
        assert len(self.obs_feature_names)+self.num_units == len(self.mine_feats_names) + len(self.ally_feats_names) + len(self.enemy_feats_names)
        
        self.hp_delta = jnp.zeros((self.n_agents, self.num_units, 5), dtype=jnp.float32)
        self.last_hp = jnp.zeros((self.n_agents, self.num_units), dtype=jnp.float32)  # 각 개체의 마지막 HP 값

        self.avail = Avail(self)

    def _extract_health(self):
        # 나 (mine), 아군 (ally), 적군 (enemy)의 hp 업데이트
        indices = {
            "mine": self.obs_feature_names.index("own_health"),
            "ally": [self.obs_feature_names.index(f"ally_health_{i+1}") for i in range(self.n_allies)],
            "enemy": [self.obs_feature_names.index(f"enemy_health_{i}") for i in range(self.n_enemies)]
        }
        mine_health = self.obs_total[:, indices["mine"]]
        ally_health = self.obs_total[:, indices["ally"]]
        enemy_health = self.obs_total[:, indices["enemy"]]
        return mine_health, ally_health, enemy_health

    def update_hp(self):
        """Wraps the JIT-compiled update_hp function."""
        
        mine_health, ally_health, enemy_health = self._extract_health()
        
        # 모든 HP 업데이트를 벡터화하여 한 번에 처리
        all_health = jnp.concatenate([
            mine_health[..., jnp.newaxis],  # 나의 HP (n_agents, 1)
            ally_health,  # 아군의 HP (n_agents, n_allies)
            enemy_health  # 적군의 HP (n_agents, n_enemies)
        ], axis=-1)  # (n_agents, 1+n_allies+n_enemies)
        
        self.hp_delta, self.last_hp = _jit_update_hp_vmap(all_health, self.hp_delta, self.last_hp, self.last_hp)

    def get_obs(self):
        """Wraps the JIT-compiled get_obs function.

        가공전 obs 정보를 활용"""
        
        self.obs_total = jnp.array(self.env_core.get_obs(), dtype=jnp.float32)
        assert self.obs_total.shape == (self.n_agents, self.obs_size)
        
        obs_mine = self.get_obs_mine(self.obs_total)
        obs_ally = self.get_obs_ally(self.obs_total)
        obs_enemy = self.get_obs_enemy(self.obs_total)
        
        self.update_hp()
        
        assert obs_mine.shape == (self.n_agents, len(self.mine_feats_names))
        assert obs_ally.shape == (self.n_agents, len(self.ally_feats_names))
        assert obs_enemy.shape == (self.n_agents, len(self.enemy_feats_names))
        
        obs_mine, obs_ally, obs_enemy = _jit_get_obs_vmap(
            obs_mine,
            obs_ally,
            obs_enemy,
            self.hp_delta,
            self.n_allies
            )
        
        return {
            "obs_mine": obs_mine,
            "obs_ally": obs_ally,
            "obs_enemy": obs_enemy,
        }

    def get_obs_mine(self, obs_total):
        """주의) 하드 코드 성향이 존재
        move-feat -> enemy-feat -> ally-feat -> own-feat
        순으로 obs_total (2d-ndarray) 가 구성됨. 이를 슬라이싱
        """
        
        mine_move_feat = obs_total[:, :self.move_feats_dim]
        mine_own_feat = obs_total[:, -self.own_feats_dim:]
        return jnp.hstack((mine_move_feat, mine_own_feat))

    def get_obs_enemy(self, obs_total):
        """주의) 하드 코드 성향이 존재
        move-feat -> enemy-feat -> ally-feat -> own-feat
        순으로 obs_total (2d-ndarray) 가 구성됨. 이를 슬라이싱
        """
        
        src = self.move_feats_dim
        dst = self.move_feats_dim + self.n_enemies*self.n_enemy_feats
        return jnp.array(obs_total[:, src: dst])

    def get_obs_ally(self, obs_total):
        """주의) 하드 코드 성향이 존재
        move-feat -> enemy-feat -> ally-feat -> own-feat
        순으로 obs_total (2d-ndarray) 가 구성됨. 이를 슬라이싱
        """
        
        src = -self.own_feats_dim - self.n_allies*self.n_ally_feats
        dst = -self.own_feats_dim
        return jnp.array(obs_total[:, src: dst])

    def get(self):
        out_dict = {
            **self.get_obs(), # obs_mine, obs_ally, obs_enemy
            **self.avail.get_avail(), # avail_act, avail_move, avail_target
        }
        return out_dict