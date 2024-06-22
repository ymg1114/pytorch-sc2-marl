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
    """주의) 개념적으로, log_prob(a) + log_prob(b) == log_prob(a, b)
    """
    
    logit_act = act_dict["logit_act"]
    logit_move = act_dict["logit_move"]
    logit_target = act_dict["logit_target"]
    
    act_sampled = act_dict["act_sampled"]
    move_sampled = act_dict["move_sampled"]
    target_sampled = act_dict["target_sampled"]
    
    on_select_act = act_dict["on_select_act"]
    on_select_move = act_dict["on_select_move"]
    on_select_target = act_dict["on_select_target"]
    
    hier_log_probs_act = cal_log_probs(logit_act, act_sampled, on_select_act)
    hier_log_probs_move = cal_log_probs(logit_move, move_sampled, on_select_move)
    hier_log_probs_target = cal_log_probs(logit_target, target_sampled, on_select_target)
    
    return hier_log_probs_act + hier_log_probs_move + hier_log_probs_target
    
    
def rew_vet_to_scaled_scalar(rew_dict):
    rew_vec = rew_dict["rew_vec"]
    
    B, S, D = rew_vec.shape
    assert D == len(REWARD_PARAM)
    
    for rdx, (r_parma, weight) in enumerate(REWARD_PARAM.items()):
        rew_vec[:, :, rdx] *= weight # 리워드 가중치 반영영
    return rew_vec.sum(-1).unsqueeze(-1)