import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F

from torch import Tensor
from torch.distributions import Categorical
from typing import TYPE_CHECKING, List, Dict, Tuple, Union

if TYPE_CHECKING:
    from env.env_proxy import EnvSpace

from agents.learner_module.compute_loss import Normalizier
from utils.utils import *


Symlog = Normalizier.symlog


def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN values found in {name}")
        print(tensor)
        
        
@jit.ignore
def build_dist(logit, avail):
    assert logit.shape == avail.shape
    if isinstance(avail, np.ndarray):
        avail = torch.tensor(avail, dtype=logit.dtype, device=logit.device)
    
    _avail = torch.where(avail == 0, torch.tensor(-1e16, dtype=logit.dtype, device=logit.device), torch.tensor(1.0, dtype=logit.dtype, device=logit.device))
    masked_logit = logit + _avail
    probs = F.softmax(masked_logit, dim=-1)
    return Categorical(probs)


class ModelSingle(nn.Module):
    def __init__(self, args, env_space: "EnvSpace"):
        super().__init__()
        # network dimension setting
        self.hidden_size = args.hidden_size
        self.dim_set(env_space)
        
        # encode
        self.encode_mine = nn.Linear(int(self.obs_mine_shape[-1]), self.hidden_size)
        self.encode_ally = nn.Linear(int(self.obs_ally_shape[-1]), self.hidden_size)
        self.encode_enemy = nn.Linear(int(self.obs_enemy_shape[-1]), self.hidden_size)
        self.encode_body = nn.Linear(self.hidden_size*3, self.hidden_size)
        self.encode_attn = nn.Linear(self.hidden_size, self.hidden_size)
        
        # normalization layers
        self.norm_mine = nn.LayerNorm(self.hidden_size)
        # self.norm_ally = nn.LayerNorm(self.hidden_size)
        # self.norm_enemy = nn.LayerNorm(self.hidden_size)
        self.norm_body = nn.LayerNorm(self.hidden_size)
        self.norm_attn = nn.LayerNorm(self.hidden_size)
        
        # # attention
        # self.mine_attn_v = nn.Linear(int(self.obs_mine_shape[-1],) self.hidden_size)
        # self.multihead_attn = nn.MultiheadAttention(self.hidden_size, num_heads=3, batch_first=True)

        # lstm
        self.lstmcell = nn.LSTMCell(self.hidden_size, self.hidden_size)

        # value
        self.num_bins = 50
        self.value = nn.Linear(self.hidden_size, self.num_bins)
        self.bins = torch.linspace(-20, 20, self.num_bins)  # 50 bins 
        
        # policy
        self.logit_act = nn.Linear(self.hidden_size, int(self.logit_act_shape[-1]))
        self.logit_move = nn.Linear(self.hidden_size, int(self.logit_move_shape[-1]))
        self.logit_target = nn.Linear(self.hidden_size, int(self.logit_target_shape[-1]))

    @staticmethod
    @jit.ignore
    def set_model_weight(args, device="cpu"):
        model_files = list(Path(args.model_dir).glob(f"{args.algo}_*.pt"))

        prev_model = None
        if len(model_files) > 0:
            sorted_files = sorted(model_files, key=extract_file_num)
            if sorted_files:
                prev_model = torch.load(
                    sorted_files[-1],
                    map_location=torch.device("cpu"),  # 가장 최신 학습 모델 cpu 텐서로 로드
                )

        if prev_model is not None:
            out_dict = {
                "state_dict": {k: v.to(device) for k, v in prev_model["model_state"].items()},
                "log_idx": prev_model["log_idx"],
                "scale": prev_model["scale"],
                "optim_state_dict": prev_model["optim_state_dict"],
            }
            return out_dict
        return
    
    @jit.ignore
    def dim_set(self, env_space):
        self.env_space = env_space
        
        obs_space = env_space["obs"]
        act_space = env_space["act"]
        # rew_space = env_space["rew"]
        # info_space = env_space["info"]
    
        self.obs_mine_shape = obs_space["obs_mine"].nvec
        self.obs_ally_shape = obs_space["obs_ally"].nvec
        self.obs_enemy_shape = obs_space["obs_enemy"].nvec
    
        self.logit_act_shape = act_space["logit_act"].nvec
        self.logit_move_shape = act_space["logit_move"].nvec
        self.logit_target_shape = act_space["logit_target"].nvec

    @jit.ignore
    def dict_ordering(
        self, item_dict: Dict[str, Union[np.ndarray, Tensor]], is_act: bool = False
    ):
        if is_act: # act-dict
            order_str = ["on_select_act", "on_select_move", "on_select_target", "act_sampled", "move_sampled", "target_sampled"]

        else: # obs-dict
            order_str = ["obs_mine", "obs_ally", "obs_enemy", "avail_act", "avail_move", "avail_target"]

        return tuple(to_torch(item_dict[key]) for key in order_str if key in item_dict)
    
    def body_encode(self, obs_tuple: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        obs_mine, obs_ally, obs_enemy = obs_tuple
        sym_obs_mine = Symlog(obs_mine)
        sym_obs_ally = Symlog(obs_ally)
        sym_obs_enemy = Symlog(obs_enemy)
        
        # # 죽은 유닛에 대한 마스크 생성, obs_* -> all zero value array
        # dead_units_mask = (sym_obs_mine.sum(dim=-1) == 0) & (sym_obs_ally.sum(dim=-1) == 0) & (sym_obs_enemy.sum(dim=-1) == 0)

        ec_mine = self.norm_mine(F.relu(self.encode_mine(sym_obs_mine)))
        ec_ally = self.encode_ally(sym_obs_ally)
        ec_enemy = self.encode_enemy(sym_obs_enemy)

        # mine_v = F.relu(self.mine_attn_v(sym_obs_mine))
        ec_body = self.norm_body(
            F.relu(self.encode_body(torch.cat([ec_mine, ec_ally, ec_enemy], dim=-1)))
        )

        # attn_out, attn_weights = self.multihead_attn(ec_mine, ec_body, mine_v, key_padding_mask=dead_units_mask.float()) # q, k, v
        # return attn_out, attn_weights
        return self.norm_attn(F.relu(self.encode_attn(ec_body)))

    @jit.ignore
    def act(
        self, obs_dict: Dict[str, np.ndarray], hx: Tensor, cx: Tensor
    ) -> Dict[str, Tensor]:
        obs_tuple = self.dict_ordering(obs_dict)
        return self._act(obs_tuple, hx, cx)

    @jit.export
    def _act(
        self,
        obs_tuple: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        hx: Tensor,
        cx: Tensor,
    ) -> Dict[str, Tensor]:
        out_encode = self.body_encode(obs_tuple[:3])
        hx, cx = self.lstmcell(out_encode, (hx, cx))

        logit_act = self.logit_act(hx)
        logit_move = self.logit_move(hx)
        logit_target = self.logit_target(hx)

        act_outs = ModelSingle.get_act_outs(
            logit_act, logit_move, logit_target, obs_tuple[3:]
        )

        out_dict = {
            "act_sampled": act_outs[0].unsqueeze(-1),
            "move_sampled": act_outs[1].unsqueeze(-1),
            "target_sampled": act_outs[2].unsqueeze(-1),
            "logit_act": act_outs[3],
            "logit_move": act_outs[4],
            "logit_target": act_outs[5],
            "hx": hx.detach(),
            "cx": cx.detach(),
        }
        return out_dict

    @staticmethod
    @jit.ignore
    def get_act_outs(
        logit_act: Tensor,
        logit_move: Tensor,
        logit_target: Tensor,
        avail_tuple: Tuple[Tensor, Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        avail_act, avail_move, avail_target = avail_tuple

        dist_act = build_dist(logit_act, avail_act)
        dist_move = build_dist(logit_move, avail_move)
        dist_target = build_dist(logit_target, avail_target)

        act_sampled, move_sampled, target_sampled = (
            dist_act.sample().detach(),
            dist_move.sample().detach(),
            dist_target.sample().detach(),
        )
        return (
            act_sampled,
            move_sampled,
            target_sampled,
            dist_act.logits.detach(),
            dist_move.logits.detach(),
            dist_target.logits.detach(),
        )

    @jit.ignore
    def forward(
        self,
        obs_dict: Dict[str, Tensor],
        act_dict: Dict[str, Tensor],
        hx: Tensor,
        cx: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        obs_tuple = self.dict_ordering(obs_dict)
        act_tuple = self.dict_ordering(act_dict, is_act=True)
        return self._forward(obs_tuple, act_tuple, hx, cx)

    @jit.export
    def _forward(
        self,
        obs_tuple: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        act_tuple: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        hx: Tensor,
        cx: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        on_select_act, on_select_move, on_select_target = act_tuple[:3]
        act_sampled, move_sampled, target_sampled = act_tuple[-3:]

        out_encode = self.body_encode(obs_tuple[:3])
        B, S, _ = out_encode.shape

        output = []
        for i in range(S):
            hx, cx = self.lstmcell(out_encode[:, i], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=1)

        value = self.value(output)

        logit_act = self.logit_act(output)
        logit_move = self.logit_move(output)
        logit_target = self.logit_target(output)

        log_probs, entropy = ModelSingle.get_forward_outs(
            logit_act,
            logit_move,
            logit_target,
            obs_tuple[3:],
            on_select_act,
            on_select_move,
            on_select_target,
            act_sampled,
            move_sampled,
            target_sampled,
        )
        return log_probs.view(B, S, -1), entropy.view(B, S, -1), value.view(B, S, -1)

    @staticmethod
    @jit.ignore
    def get_forward_outs(
        logit_act: Tensor,
        logit_move: Tensor,
        logit_target: Tensor,
        avail_tuple: Tuple[Tensor, Tensor, Tensor],
        on_select_act: Tensor,
        on_select_move: Tensor,
        on_select_target: Tensor,
        act_sampled: Tensor,
        move_sampled: Tensor,
        target_sampled: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        dist_act, dist_move, dist_target = [
            build_dist(logit, avail)
            for logit, avail in zip([logit_act, logit_move, logit_target], avail_tuple)
        ]

        log_probs = torch.stack(
            [
                on_select * dist.log_prob(sampled.squeeze(-1)).unsqueeze(-1)
                for on_select, dist, sampled in zip(
                    [on_select_act, on_select_move, on_select_target],
                    [dist_act, dist_move, dist_target],
                    [act_sampled, move_sampled, target_sampled],
                )
            ],
            dim=-1,
        ).sum(-1)

        entropy = torch.stack(
            [
                on_select * dist.entropy().unsqueeze(-1)
                for on_select, dist in zip(
                    [on_select_act, on_select_move, on_select_target],
                    [dist_act, dist_move, dist_target],
                )
            ],
            dim=-1,
        ).sum(-1)

        return log_probs, entropy