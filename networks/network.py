import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.env_proxy import EnvSpace
    
from utils.utils import *


def build_dist(logit, avail):
    assert logit.shape == avail.shape
    _avail = avail.clone() # 레퍼런스 오염 방지
    _avail[_avail == 0] = -1e+7 # 선택 불가
    _avail[_avail == 1] = 1.0 # 선택 가능

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

        # attention
        self.mine_attn_v = nn.Linear(self.obs_mine_shape[-1], self.hidden_size)
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, num_heads=2, batch_first=True)

        # lstm
        self.lstmcell = nn.LSTMCell(self.hidden_size, self.hidden_size)

        # value
        self.value = nn.Linear(self.hidden_size, 1)

        # policy
        self.logit_act = nn.Linear(self.hidden_size, self.logit_act_shape[-1])
        self.logit_move = nn.Linear(self.hidden_size, self.logit_move_shape[-1])
        self.logit_target = nn.Linear(self.hidden_size, self.logit_target_shape[-1])

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

        obs_mine = obs_dict["obs_mine"]
        obs_ally = obs_dict["obs_ally"]
        obs_enemy = obs_dict["obs_enemy"]

        # 죽은 유닛에 대한 마스크 생성, obs_* -> all zero value array
        dead_units_mask = (obs_mine.sum(dim=-1) == 0) & (obs_ally.sum(dim=-1) == 0) & (obs_enemy.sum(dim=-1) == 0)

        ec_mine = F.relu(self.encode_mine(obs_mine))
        ec_ally = F.relu(self.encode_ally(obs_ally))
        ec_enemy = F.relu(self.encode_enemy(obs_enemy))

        mine_v = F.relu(self.mine_attn_v(obs_mine))

        ec_body = F.relu(self.encode_body(torch.cat([ec_mine, ec_ally, ec_enemy], dim=-1)))
        
        return self.multihead_attn(ec_mine, ec_body, mine_v, key_padding_mask=dead_units_mask.bool()) # q, k, v
    
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

    def act(self, obs_dict, lstm_hxs):
        attn_out, attn_weights = self.body_encode(obs_dict)
        
        hx, cx = self.lstmcell(attn_out, lstm_hxs)

        dist_act, dist_move, dist_target = self.get_dists(hx, obs_dict)

        act_sampled = dist_act.sample().detach()
        move_sampled = dist_move.sample().detach()
        target_sampled = dist_target.sample().detach()

        out_dict = {
            "act_sampled": act_sampled,
            "move_sampled": move_sampled,
            "target_sampled": target_sampled,
            "logit_act": dist_act.logits.detach(),
            "logit_move": dist_move.logits.detach(),
            "logit_target": dist_target.logits.detach(),
            # "log_prob_act": dist_act.log_prob(act_sampled).detach(),
            # "log_prob_move": dist_move.log_prob(move_sampled).detach(),
            # "log_prob_target": dist_target.log_prob(target_sampled).detach(),
            "lstm_hxs": (hx.detach(), cx.detach()),
        }
        return out_dict

    def forward(self, obs_dict, act_dict, lstm_hxs):
        on_select_act = act_dict["on_select_act"]
        on_select_move = act_dict["on_select_move"]
        on_select_target = act_dict["on_select_target"]
        
        attn_out, attn_weights = self.body_encode(obs_dict)
        B, S, _ = attn_out.shape
        
        hx, cx = lstm_hxs
        output = []
        for i in range(S):
            hx, cx = self.lstmcell(attn_out[:, i], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=1)

        value = self.value(output)
        
        dist_act, dist_move, dist_target = self.get_dists(output, obs_dict)

        act_sampled = dist_act.sample().detach()
        move_sampled = dist_move.sample().detach()
        target_sampled = dist_target.sample().detach()

        log_probs = 0.0
        log_probs += on_select_act * dist_act.log_prob(act_sampled.squeeze(-1)).unsqueeze(-1)
        log_probs += on_select_move * dist_move.log_prob(move_sampled.squeeze(-1)).unsqueeze(-1)
        log_probs += on_select_target * dist_target.log_prob(target_sampled.squeeze(-1)).unsqueeze(-1)

        entropy = 0.0
        entropy += on_select_act * dist_act.entropy()
        entropy += on_select_move * dist_move.entropy()
        entropy += on_select_target * dist_target.entropy()

        return (
            log_probs.view(B, S, -1),
            entropy.view(B, S, -1),
            value.view(B, S, -1),
        )