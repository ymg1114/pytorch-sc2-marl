import torch
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

from rewarder.rewarder import REWARD_PARAM


def compute_gae(
    deltas,
    gamma,
    lambda_,
):
    gae = 0
    returns = []
    for t in reversed(range(deltas.size(1))):
        d = deltas[:, t]
        gae = d + gamma * lambda_ * gae
        returns.insert(0, gae)

    return torch.stack(returns, dim=1)


def compute_v_trace(
    behav_log_probs,
    target_log_probs,
    is_fir,
    rewards,
    values,
    gamma,
    rho_bar=0.8,
    c_bar=1.0,
):
    # Importance sampling weights (rho)
    rho = torch.exp(
        target_log_probs[:, :-1] - behav_log_probs[:, :-1]
    )  # a/b == exp(log(a)-log(b))
    # rho_clipped = torch.clamp(rho, max=rho_bar)
    rho_clipped = torch.clamp(rho, min=0.1, max=rho_bar)

    # truncated importance weights (c)
    c = torch.exp(
        target_log_probs[:, :-1] - behav_log_probs[:, :-1]
    )  # a/b == exp(log(a)-log(b))
    c_clipped = torch.clamp(c, max=c_bar)

    td_target = rewards[:, :-1] + gamma * (1 - is_fir[:, 1:]) * values[:, 1:]
    deltas = rho_clipped * (td_target - values[:, :-1])  # TD-Error with 보정

    vs_minus_v_xs = torch.zeros_like(values)
    for t in reversed(range(deltas.size(1))):
        vs_minus_v_xs[:, t] = (
            deltas[:, t]
            + c_clipped[:, t]
            * (gamma * (1 - is_fir))[:, t + 1]
            * vs_minus_v_xs[:, t + 1]
        )

    values_target = (
        values + vs_minus_v_xs
    )  # vs_minus_v_xs는 V-trace를 통해 수정된 가치 추정치
    advantages = rho_clipped * (
        rewards[:, :-1]
        + gamma * (1 - is_fir[:, 1:]) * values_target[:, 1:]
        - values[:, :-1]
    )

    return rho_clipped, advantages, values_target


def kldivergence(logits_p, logits_q):
    dist_p = Categorical(F.softmax(logits_p, dim=-1))
    dist_q = Categorical(F.softmax(logits_q, dim=-1))
    return kl_divergence(dist_p, dist_q).squeeze()


def cal_log_probs(logit, sampled, on_select):
    return on_select * Categorical(F.softmax(logit, dim=-1)).log_prob(sampled.squeeze(-1)).unsqueeze(-1)


def cal_hier_log_probs(act_dict):
    """개념적으로, log_prob(a) + log_prob(b) == log_prob(a, b)를 사용하여 계층적 log 확률을 연산."""
    
    keys = ["logit_act", "logit_move", "logit_target"]
    sampled_keys = ["act_sampled", "move_sampled", "target_sampled"]
    select_keys = ["on_select_act", "on_select_move", "on_select_target"]

    hier_log_probs = sum(
        cal_log_probs(act_dict[logit_key], act_dict[sampled_key], act_dict[select_key])
        for logit_key, sampled_key, select_key in zip(keys, sampled_keys, select_keys)
    )
    
    return hier_log_probs


def rew_vec_to_scaled_scalar(rew_dict):
    rew_vec = rew_dict["rew_vec"]
    
    B, S, D = rew_vec.shape
    assert D == len(REWARD_PARAM)

    weights = torch.tensor(list(REWARD_PARAM.values()), dtype=rew_vec.dtype, device=rew_vec.device)
    scaled_rew_vec = rew_vec * weights
    return scaled_rew_vec.sum(-1, keepdim=True)