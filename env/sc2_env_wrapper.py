# import torch
import numpy as np

from smacv2.env import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from pysc2.lib import protocol
from absl import logging

from rewarder.rewarder import Rewarder, REWARD_PARAM
from observer.observer import Observer

# from gymnasium.spaces import Dict
from env.env_proxy import EnvSpace, ObservationSpace, ActionSpace, RewardSpace, InfoSpace, OthersSpace, MultiDiscreteSpaceDim as MDSD, TextSpaceDim as TSD

from utils.utils import *

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb


class WrapperSC2Env(StarCraft2Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: 임시, FLEE 행동 대처
        self.n_actions += 1
        
        self.rewarder = Rewarder(self)
        self.observer = Observer(self)

    def act_dict_converter(self, act_dict):
        act_sampled = act_dict["act_sampled"].numpy()
        move_sampled = act_dict["move_sampled"].numpy()
        target_sampled = act_dict["target_sampled"].numpy()
        assert act_sampled.shape == move_sampled.shape == target_sampled.shape
        assert self.n_actions_no_attack == 6
        
        # 초기화
        actions = np.zeros_like(act_sampled)

        # NO_OP, STOP 행동은 그대로 가져감
        act_idx = np.where((act_sampled == NO_OP_IDX) | (act_sampled == STOP_IDX))
        actions[act_idx] = act_sampled[act_idx]

        # MOVE 행동은 해당 움직임의 방향 인덱스로 변경
        # NO_OP, STOP 2개 행동 이후부터 4개 -> MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST
        move_idx = np.where(act_sampled == MOVE_IDX)
        act_dict["on_select_move"][move_idx] = 1.0 # 선택했음을 알려줌
        actions[move_idx] = move_sampled[move_idx] + 2 # 실제 행동 인덱스에 맞추기 위해 shift

        # TARGET 행동은 해당 유닛 인덱스로 변경
        # NO_OP, STOP, MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST 6개 행동 이후부터 나머지
        tar_idx = np.where(act_sampled == TARGET_IDX)
        act_dict["on_select_target"][tar_idx] = 1.0 # 선택했음을 알려줌
        actions[tar_idx] = target_sampled[tar_idx] + self.n_actions_no_attack  # 실제 행동 인덱스에 맞추기 위해 shift

        # TODO: FLEE 행동은 -1번 인덱스 행동으로 임의로 맵핑 (신경망 logits 레벨에서는 4번)
        flee_idx = np.where(act_sampled == FLEE_IDX)
        actions[flee_idx] = -1
        
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
                self.marine_id: 9, # TODO 이 값으로 거리 개념의 feat를 나눠 버림 (Normalize)
                self.marauder_id: 10,
                self.medivac_id: 11,
                1970: 9, # TODO: 버그 대응. 유닛 타입 1970 (마린)의 시야 범위를 추가 (필요에 따라 수정)
            }
            unit = self.agents[agent_id]
            return sight_range_map[unit.unit_type]
        else:
            return 9

    def get_agent_action(self, a_id, action):
        """TODO: FLEE 행동을 -1번 인덱스에 맵핑하고, 나머지 기존 행동들은 SC2 환경에서 제공하는 것을 그대로 사용하기 위함
        """
        
        if action == -1:
            unit = self.get_unit_by_id(a_id)
            # tag = unit.tag
            # x = unit.pos.x
            # y = unit.pos.y

            flee_target_y = self.observer.avail.flee_positions_np["y"][a_id]
            flee_target_x = self.observer.avail.flee_positions_np["x"][a_id]

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=16,  # TODO: 하드코드, "move" / target: PointOrUnit
                target_world_space_pos=sc_common.Point2D(
                    x=flee_target_x, 
                    y=flee_target_y
                ),
                unit_tags=[unit.tag],
                queue_command=False,
            )

            self.new_unit_positions[a_id] = np.array(
                [flee_target_x, flee_target_y]
            )

            sc_action = sc_pb.Action(
                action_raw=r_pb.ActionRaw(unit_command=cmd)
            )
            return sc_action
        else:
            return super().get_agent_action(a_id, action)
        
    def step_dict(self, act_dict, dead_agents_vec):
        """A single environment step. Returns reward, terminated, info."""
        
        #TODO: 임시
        assert self.heuristic_ai == False, f"일단 이 경우만 다룬다. heuristic_ai: {self.heuristic_ai}"
        
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
                # actions[a_id] = action_num # TODO: 필요한 처리 로직..?
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
        info = {}
        # info = {"battle_won": False}
        
        # 각 유닛 (에이전트) 별 리워드 획득
        reward_vec = self.rewarder.get(game_end_code)

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for aid, (_al_id, al_unit) in enumerate(self.agents.items()):
            if al_unit.health == 0:
                dead_allies += 1
                dead_agents_vec[aid] = 1.0 # next-state 기준 죽은 에이전트
                
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
                # info["battle_won"] = True
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
            
        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            logging.debug("Reward = {}".format(reward_vec).center(60, "-"))

        if terminated:
            self._episode_count += 1

        # TODO: 원래 코드가 이렇게 사용하는데, 이 부분에서 버그가 발생할 수 있음. scalar -> vector
        # self.reward = reward
        self.reward_vec = reward_vec

        return reward_vec, terminated, info


class WrapperSMAC2(StarCraftCapabilityEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = WrapperSC2Env(*args, **kwargs) # override

    # def update_death_tracker(self):
    #     """리워드 연산 후, 자료형 업데이트
    #     그다음 "step" 생명 주기를 호출하기 전에 반드시 수행해야 함
    #     """

    #     for al_id, al_unit in self.env.agents.items():
    #         if not self.env.death_tracker_ally[al_id]: # did not die so far
    #             if al_unit.health == 0:
    #                 # just died
    #                 self.env.death_tracker_ally[al_id] = 1 # 자료형 업데이트

    #     for e_id, e_unit in self.env.enemies.items():
    #         if not self.env.death_tracker_enemy[e_id]: # did not die so far
    #             if e_unit.health == 0:
    #                 # just died
    #                 self.env.death_tracker_enemy[e_id] = 1 # 자료형 업데이트

    def get_death_tracker_ally(self):
        return self.env.death_tracker_ally

    def get_obs_dict(self):
        return self.env.observer.get()

    def get_env_space(self, args):
        _obs = self.env.observer
        # _avail = _obs.avail
        B = args.batch_size
        S = args.seq_len

        env_space = EnvSpace(
            obs=ObservationSpace(
                obs_mine=MDSD(nvec=(B, S, len(_obs.mine_feats_names))),
                obs_ally=MDSD(nvec=(B, S, len(_obs.ally_feats_names))),
                obs_enemy=MDSD(nvec=(B, S, len(_obs.enemy_feats_names))),
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
                rew_vec=MDSD(nvec=(B, S, len(REWARD_PARAM)))
            ),
            info=InfoSpace(
                is_fir=MDSD(nvec=(B, S, 1))
            ),
            others=OthersSpace(
                mine_feats_names=TSD(names=_obs.mine_feats_names),
                ally_feats_names=TSD(names=_obs.ally_feats_names),
                enemy_feats_names=TSD(names=_obs.enemy_feats_names),
                num_allys=_obs.n_allies,
                num_enemys=_obs.n_enemies,
            )
        )
        return env_space.to_gym_space()
