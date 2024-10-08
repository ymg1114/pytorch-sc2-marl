from typing import Union, Tuple, List, Any
from dataclasses import dataclass, fields
from gymnasium.spaces import Dict, MultiDiscrete, Text, Box


@dataclass
class SpaceBase:
    def _convert_field(self, field_value):
        if hasattr(field_value, 'to_multi_discrete'):
            return field_value.to_multi_discrete()
        elif hasattr(field_value, 'to_text'):
            return field_value.to_text()
        return field_value

    def to_dict(self):
        return {f.name: self._convert_field(getattr(self, f.name)) for f in fields(self)}


@dataclass
class TextSpaceDim:
    names: Union[List[str], Tuple[str]]

    def to_text(self):
        max_length = max(len(name) for name in self.names)  # Determine the max length of text
        return Text(max_length=max_length)


@dataclass
class MultiDiscreteSpaceDim:
    nvec: Tuple[Union[int, float]]

    def to_multi_discrete(self):
        return MultiDiscrete(self.nvec)


@dataclass
class ObservationSpace(SpaceBase):
    obs_mine: MultiDiscreteSpaceDim
    obs_ally: MultiDiscreteSpaceDim
    obs_enemy: MultiDiscreteSpaceDim
    avail_act: MultiDiscreteSpaceDim
    avail_move: MultiDiscreteSpaceDim
    avail_target: MultiDiscreteSpaceDim
    hx: MultiDiscreteSpaceDim
    cx: MultiDiscreteSpaceDim


@dataclass
class ActionSpace(SpaceBase):
    act_sampled: MultiDiscreteSpaceDim
    move_sampled: MultiDiscreteSpaceDim
    target_sampled: MultiDiscreteSpaceDim
    logit_act: MultiDiscreteSpaceDim
    logit_move: MultiDiscreteSpaceDim
    logit_target: MultiDiscreteSpaceDim
    on_select_act: MultiDiscreteSpaceDim
    on_select_move: MultiDiscreteSpaceDim
    on_select_target: MultiDiscreteSpaceDim


@dataclass
class RewardSpace(SpaceBase):
    rew_vec: MultiDiscreteSpaceDim


@dataclass
class InfoSpace(SpaceBase):
    is_fir: MultiDiscreteSpaceDim


@dataclass
class OthersSpace(SpaceBase):
    mine_feats_names: TextSpaceDim
    ally_feats_names: TextSpaceDim
    enemy_feats_names: TextSpaceDim
    num_allys: Union[int, float]
    num_enemys: Union[int, float]

    def to_dict(self):
        return {
            "mine_feats_names": self.mine_feats_names.to_text(),
            "ally_feats_names": self.ally_feats_names.to_text(),
            "enemy_feats_names": self.enemy_feats_names.to_text(),
            "num_allys": Box(low=0, high=self.num_allys, shape=(), dtype=int),
            "num_enemys": Box(low=0, high=self.num_enemys, shape=(), dtype=int),
        }


@dataclass
class EnvSpace:
    obs: ObservationSpace
    act: ActionSpace
    rew: RewardSpace
    info: InfoSpace
    others: OthersSpace

    def to_gym_space(self):
        return Dict({
            "obs": Dict(self.obs.to_dict()),
            "act": Dict(self.act.to_dict()),
            "rew": Dict(self.rew.to_dict()),
            "info": Dict(self.info.to_dict()),
            "others": Dict(self.others.to_dict()),
        })
