import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.env_proxy import EnvSpace

from agents.learner_module.compute_loss import Normalizier
from utils.utils import *


Symlog = Normalizier.symlog


def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN values found in {name}")
        print(tensor)
        

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
        self.encode_mine = nn.Linear(self.obs_mine_shape[-1], self.hidden_size)
        self.encode_ally = nn.Linear(self.obs_ally_shape[-1], self.hidden_size)
        self.encode_enemy = nn.Linear(self.obs_enemy_shape[-1], self.hidden_size)
        self.encode_body = nn.Linear(self.hidden_size*3, self.hidden_size)
        self.encode_attn = nn.Linear(self.hidden_size, self.hidden_size)
        
        # normalization layers
        self.norm_mine = nn.LayerNorm(self.hidden_size)
        # self.norm_ally = nn.LayerNorm(self.hidden_size)
        # self.norm_enemy = nn.LayerNorm(self.hidden_size)
        self.norm_body = nn.LayerNorm(self.hidden_size)
        self.norm_attn = nn.LayerNorm(self.hidden_size)
        
        # # attention
        # self.mine_attn_v = nn.Linear(self.obs_mine_shape[-1], self.hidden_size)
        # self.multihead_attn = nn.MultiheadAttention(self.hidden_size, num_heads=3, batch_first=True)

        # lstm
        self.lstmcell = nn.LSTMCell(self.hidden_size, self.hidden_size)

        # value
        self.num_bins = 50
        self.value = nn.Linear(self.hidden_size, self.num_bins)
        self.bins = torch.linspace(-20, 20, self.num_bins)  # 50 bins 
        
        # policy
        self.logit_act = nn.Linear(self.hidden_size, self.logit_act_shape[-1])
        self.logit_move = nn.Linear(self.hidden_size, self.logit_move_shape[-1])
        self.logit_target = nn.Linear(self.hidden_size, self.logit_target_shape[-1])

    @staticmethod
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

    def np_to_torch(self, obs_dict):
        return {k: to_torch(v) for k, v in obs_dict.items()}

    def body_encode(self, obs_dict):
        obs_dict = self.np_to_torch(obs_dict)

        obs_mine = Symlog(obs_dict["obs_mine"])
        obs_ally = Symlog(obs_dict["obs_ally"])
        obs_enemy = Symlog(obs_dict["obs_enemy"])

        # # 죽은 유닛에 대한 마스크 생성, obs_* -> all zero value array
        # dead_units_mask = (obs_mine.sum(dim=-1) == 0) & (obs_ally.sum(dim=-1) == 0) & (obs_enemy.sum(dim=-1) == 0)

        ec_mine = self.norm_mine(F.relu(self.encode_mine(obs_mine)))
        ec_ally = self.encode_ally(obs_ally)
        ec_enemy = self.encode_enemy(obs_enemy)

        # mine_v = F.relu(self.mine_attn_v(obs_mine))

        ec_body = self.norm_body(F.relu(self.encode_body(torch.cat([ec_mine, ec_ally, ec_enemy], dim=-1))))

        # attn_out, attn_weights = self.multihead_attn(ec_mine, ec_body, mine_v, key_padding_mask=dead_units_mask.float()) # q, k, v
        # return attn_out, attn_weights
        out_encode = self.norm_attn(F.relu(self.encode_attn(ec_body)))
        return out_encode
        
    def get_dists(self, x, obs_dict):
        avail_act = obs_dict["avail_act"]
        avail_move = obs_dict["avail_move"]
        avail_target = obs_dict["avail_target"]

        logit_act = self.logit_act(x)
        logit_move = self.logit_move(x)
        logit_target = self.logit_target(x)

        dist_act = build_dist(logit_act, avail_act)
        dist_move = build_dist(logit_move, avail_move)
        dist_target = build_dist(logit_target, avail_target)

        return dist_act, dist_move, dist_target

    def act(self, obs_dict, hx, cx):
        out_encode = self.body_encode(obs_dict)
        
        hx, cx = self.lstmcell(out_encode, (hx, cx))

        dist_act, dist_move, dist_target = self.get_dists(hx, obs_dict)

        act_sampled = dist_act.sample().detach()
        move_sampled = dist_move.sample().detach()
        target_sampled = dist_target.sample().detach()

        out_dict = {
            "act_sampled": act_sampled.unsqueeze(-1),
            "move_sampled": move_sampled.unsqueeze(-1),
            "target_sampled": target_sampled.unsqueeze(-1),
            "logit_act": dist_act.logits.detach(),
            "logit_move": dist_move.logits.detach(),
            "logit_target": dist_target.logits.detach(),
            "hx": hx.detach(),
            "cx": cx.detach(),
        }
        return out_dict

    def forward(self, obs_dict, act_dict, hx, cx):
        on_select_act = act_dict["on_select_act"]
        on_select_move = act_dict["on_select_move"]
        on_select_target = act_dict["on_select_target"]
        
        act_sampled = act_dict["act_sampled"]
        move_sampled = act_dict["move_sampled"]
        target_sampled = act_dict["target_sampled"]
        
        out_encode = self.body_encode(obs_dict)
        B, S, _ = out_encode.shape
        
        output = []
        for i in range(S):
            hx, cx = self.lstmcell(out_encode[:, i], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=1)

        value = self.value(output)
        dist_act, dist_move, dist_target = self.get_dists(output, obs_dict)

        log_probs = sum(
            on_select * dist.log_prob(sampled.squeeze(-1)).unsqueeze(-1)
            for on_select, dist, sampled in zip(
                [on_select_act, on_select_move, on_select_target],
                [dist_act, dist_move, dist_target],
                [act_sampled, move_sampled, target_sampled]
            )
        )
        
        entropy = sum(
            on_select * dist.entropy().unsqueeze(-1)
            for on_select, dist in zip(
                [on_select_act, on_select_move, on_select_target],
                [dist_act, dist_move, dist_target]
            )
        )
        return log_probs.view(B, S, -1), entropy.view(B, S, -1), value.view(B, S, -1)