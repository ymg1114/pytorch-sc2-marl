from dataclasses import dataclass, field, fields
from gymnasium.spaces import Dict, MultiDiscrete


@dataclass
class MultiDiscreteSpaceDim:
    nvec: tuple

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
class EnvSpace:
    obs: ObservationSpace
    act: ActionSpace
    rew: RewardSpace
    info: InfoSpace

    def to_gym_space(self):
        return Dict({
            "obs": Dict(self.obs.to_dict()),
            "act": Dict(self.act.to_dict()),
            "rew": Dict(self.rew.to_dict()),
            "info": Dict(self.info.to_dict())
        })