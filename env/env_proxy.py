from typing import Union, Tuple, List
from dataclasses import dataclass, field, fields
from gymnasium.spaces import Dict, MultiDiscrete, Text


@dataclass
class TextSpaceDim:
    names: Union[List[str], Tuple[str]]

    def to_text(self):
        # max_length = max(len(name) for name in self.names) if self.names else 1
        # return Text(max_length=max_length)
        return self.names


@dataclass
class MultiDiscreteSpaceDim:
    nvec: Tuple[Union[int, float]]

    def to_multi_discrete(self):
        return MultiDiscrete(self.nvec)


@dataclass
class ObservationSpace:
    obs_mine: MultiDiscreteSpaceDim
    obs_ally: MultiDiscreteSpaceDim
    obs_enemy: MultiDiscreteSpaceDim
    avail_act: MultiDiscreteSpaceDim
    avail_move: MultiDiscreteSpaceDim
    avail_target: MultiDiscreteSpaceDim
    hx: MultiDiscreteSpaceDim
    cx: MultiDiscreteSpaceDim

    def to_dict(self):
        return {f.name: getattr(self, f.name).to_multi_discrete() for f in fields(self)}


@dataclass
class ActionSpace:
    act_sampled: MultiDiscreteSpaceDim
    move_sampled: MultiDiscreteSpaceDim
    target_sampled: MultiDiscreteSpaceDim
    logit_act: MultiDiscreteSpaceDim
    logit_move: MultiDiscreteSpaceDim
    logit_target: MultiDiscreteSpaceDim
    on_select_act: MultiDiscreteSpaceDim
    on_select_move: MultiDiscreteSpaceDim
    on_select_target: MultiDiscreteSpaceDim

    def to_dict(self):
        return {f.name: getattr(self, f.name).to_multi_discrete() for f in fields(self)}


@dataclass
class RewardSpace:
    rew_vec: MultiDiscreteSpaceDim

    def to_dict(self):
        return {f.name: getattr(self, f.name).to_multi_discrete() for f in fields(self)}


@dataclass
class InfoSpace:
    is_fir: MultiDiscreteSpaceDim

    def to_dict(self):
        return {f.name: getattr(self, f.name).to_multi_discrete() for f in fields(self)}


@dataclass
class OthersSpace:
    mine_feats_names: TextSpaceDim
    ally_feats_names: TextSpaceDim
    enemy_feats_names: TextSpaceDim
    num_allys: Union[int, float]
    num_enemys: Union[int, float]

    def _convert_field(self, field_value):
        return field_value.to_text() if hasattr(field_value, 'to_text') else field_value

    def to_dict(self):
        return {f.name: self._convert_field(getattr(self, f.name)) for f in fields(self)}


@dataclass
class EnvSpace:
    obs: ObservationSpace
    act: ActionSpace
    rew: RewardSpace
    info: InfoSpace
    others: OthersSpace
    
    #TODO: 일단.. 버그 대응하기 위해, 아래와 같이 임시로 처리
    def to_gym_space(self):
        # return Dict({
        #     "obs": Dict(self.obs.to_dict()),
        #     "act": Dict(self.act.to_dict()),
        #     "rew": Dict(self.rew.to_dict()),
        #     "info": Dict(self.info.to_dict()),
        #     "others": self.others.to_dict(),
        # })
        return {
            "obs": Dict(self.obs.to_dict()),
            "act": Dict(self.act.to_dict()),
            "rew": Dict(self.rew.to_dict()),
            "info": Dict(self.info.to_dict()),
            "others": self.others.to_dict(),
        }