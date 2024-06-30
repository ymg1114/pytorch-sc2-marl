# import torch
import numpy as np

from smacv2.env import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from pysc2.lib import protocol
from absl import logging

from s2clientprotocol import sc2api_pb2 as sc_pb

# from rewarder.rewarder import Rewarder, REWARD_PARAM
from observer.observer import Observer

# from gymnasium.spaces import Dict
from env.env_proxy import EnvSpace, ObservationSpace, ActionSpace, RewardSpace, InfoSpace, MultiDiscreteSpaceDim as MDSD

from utils.utils import *


class WrapperSC2Env(StarCraft2Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.rewarder = Rewarder(self)
        self.observer = Observer(self)

    def act_dict_converter(self, act_dict):
        act_sampled = act_dict["act_sampled"].numpy()
        move_sampled = act_dict["move_sampled"].numpy()
        target_sampled = act_dict["target_sampled"].numpy()
        assert act_sampled.shape == move_sampled.shape == target_sampled.shape

        # 초기화
        actions = np.zeros_like(act_sampled)

        # NO_OP, STOP 행동은 그대로 가져감
        act_idx = np.where((act_sampled == NO_OP_IDX) | (act_sampled == STOP_IDX))
        actions[act_idx] = act_sampled[act_idx]

        # MOVE 행동은 해당 움직임의 방향 인덱스로 변경
        # NO_OP, STOP 2개 행동 이후부터 4개 -> MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST
        move_idx = np.where(act_sampled == MOVE_IDX)
        act_dict["on_select_move"][move_idx] = 1.0
        actions[move_idx] = move_sampled[move_idx] + 2 # 실제 행동 인덱스에 맞추기 위해 shift

        # TARGET 행동은 해당 유닛 인덱스로 변경
        # NO_OP, STOP, MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST 6개 행동 이후부터 나머지
        tar_idx = np.where(act_sampled == TARGET_IDX)
        act_dict["on_select_target"][tar_idx] = 1.0
        actions[tar_idx] = target_sampled[tar_idx] + 6 # 실제 행동 인덱스에 맞추기 위해 shift

        return actions

    def unit_shoot_range(self, agent_id):
        """Returns the shooting range for an agent.
        
        주의) TODO: 원래 SMAC2 환경에서 알 수 없는..? key-error (1970) 이 발생해, 부득이하게 메서드 override
        """
        
        if self.use_unit_ranges:
            attack_range_map = {
                self.stalker_id: 6,
                self.zealot_id: 0.1,
                self.colossus_id: 7,
                self.zergling_id: 0.1,
                self.baneling_id: 0.25,
                self.hydralisk_id: 5,
                self.marine_id: 5,
                self.marauder_id: 6,
                self.medivac_id: 4,
                1970: 5,  # TODO: 버그 대응. 유닛 타입 1970 (마린)의 유효 사거리 범위를 추가 (필요에 따라 수정)
            }
            unit = self.agents[agent_id]
            return max(attack_range_map[unit.unit_type], self.min_attack_range)
        else:
            return 6

    def unit_sight_range(self, agent_id):
        """Returns the sight range for an agent.
        
        주의) TODO: 원래 SMAC2 환경에서 알 수 없는..? key-error (1970) 이 발생해, 부득이하게 메서드 override
        """
        
        # get the unit
        if self.use_unit_ranges:
            sight_range_map = {
                self.stalker_id: 10,
                self.zealot_id: 9,
                self.colossus_id: 10,
                self.zergling_id: 8,
                self.baneling_id: 8,
                self.hydralisk_id: 9,
                self.marine_id: 9,
                self.marauder_id: 10,
                self.medivac_id: 11,
                1970: 9, # TODO: 버그 대응. 유닛 타입 1970 (마린)의 시야 범위를 추가 (필요에 따라 수정)
            }
            unit = self.agents[agent_id]
            return sight_range_map[unit.unit_type]
        else:
            return 9

    def step_dict(self, act_dict, dead_agents_vec):
        """A single environment step. Returns reward, terminated, info."""
        
        actions_int = [int(a) for a in self.act_dict_converter(act_dict)]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(actions_int):
            if not self.heuristic_ai:
                sc_action = self.get_agent_action(a_id, action)
            else:
                sc_action, action_num = self.get_agent_action_heuristic(
                    a_id, action
                )
                # actions[a_id] = action_num
            if sc_action:
                sc_actions.append(sc_action)
        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)

        try:
            if self.conic_fov:
                self.render_fovs()
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            if not self.stochastic_health:
                self._controller.step(self._step_mul)
            else:
                self._controller.step(
                    self._step_mul - self._kill_unit_step_mul
                )
                self._kill_units_below_health_level()
                self._controller.step(self._kill_unit_step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            return np.zeros((len(REWARD_PARAM), len(self.agents)), dtype=np.float32), True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {"battle_won": False}

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for aid, (_al_id, al_unit) in enumerate(self.agents.items()):
            if al_unit.health == 0:
                dead_allies += 1
                dead_agents_vec[aid] = 1.0 # 죽은 에이전트
                for e in range(self.n_enemies):
                    if self.enemy_tags[e] == _al_id:
                        self.enemy_tags[e] = None
                        self.obs_enemies[e, :] = 0
                        self.obs_enemies[:, _al_id] = 0
        for _e_id, e_unit in self.enemies.items():
            if e_unit.health == 0:
                dead_enemies += 1
                self.enemy_tags[_e_id] = None
                self.obs_enemies[_e_id, :] = 0

        info["dead_allies"] = dead_allies
        info["dead_enemies"] = dead_enemies

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, "-"))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        self.reward = reward

        return reward, terminated, info


class WrapperSMAC2(StarCraftCapabilityEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = WrapperSC2Env(*args, **kwargs) # override

    def get_obs_dict(self):
        return self.env.observer.get()

    def get_env_space(self, args):
        _obs = self.env.observer
        # _avail = _obs.avail
        B = args.batch_size
        S = args.seq_len
        
        env_space = EnvSpace(
            obs=ObservationSpace(
                obs_mine=MDSD(nvec=(B, S, _obs.move_feats_dim + _obs.own_feats_dim)),
                obs_ally=MDSD(nvec=(B, S, _obs.ally_feats_dim[0] * _obs.ally_feats_dim[1])),
                obs_enemy=MDSD(nvec=(B, S, _obs.enemy_feats_dim[0] * _obs.enemy_feats_dim[1])),
                avail_act=MDSD(nvec=(B, S, _obs.dim_act)),
                avail_move=MDSD(nvec=(B, S, _obs.dim_move)),
                avail_target=MDSD(nvec=(B, S, _obs.dim_target)),
                hx=MDSD(nvec=(B, S, args.hidden_size)),
                cx=MDSD(nvec=(B, S, args.hidden_size))
            ),
            act=ActionSpace(
                act_sampled=MDSD(nvec=(B, S, 1)),
                move_sampled=MDSD(nvec=(B, S, 1)),
                target_sampled=MDSD(nvec=(B, S, 1)),
                logit_act=MDSD(nvec=(B, S, _obs.dim_act)),
                logit_move=MDSD(nvec=(B, S, _obs.dim_move)),
                logit_target=MDSD(nvec=(B, S, _obs.dim_target)),
                on_select_act=MDSD(nvec=(B, S, 1)),
                on_select_move=MDSD(nvec=(B, S, 1)),
                on_select_target=MDSD(nvec=(B, S, 1))
            ),
            rew=RewardSpace(
                rew_vec=MDSD(nvec=(B, S, 1))
            ),
            info=InfoSpace(
                is_fir=MDSD(nvec=(B, S, 1))
            )
        )
        return env_space.to_gym_space()
