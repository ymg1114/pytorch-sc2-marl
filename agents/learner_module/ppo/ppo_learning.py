import jax
import jax.numpy as jnp
from flax import nnx

from functools import partial

from networks.network import jax_model_forward
from ..learner_lib import ALPHA, Q_HIGH, Q_LOW, HyperParams, TrainingBatch, goal_curr_alive_mine_mask, append_loss, compute_gae, cal_hier_log_probs, rew_vec_to_scaled_scalar, cross_entropy_loss, Normalizier

from typing import Tuple


Symlog = Normalizier.symlog
NormReturns = Normalizier.norm_returns
CalculateScale = Normalizier.calculate_s
TwohotDecoding = Normalizier.twohot_decoding
TwohotEncoding = Normalizier.twohot_encoding


def loss_fn(
    model: nnx.Module,
    hyperparams: HyperParams,
    training_batch: TrainingBatch
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    obs_dict = training_batch.obs_dict
    act_dict = training_batch.act_dict
    rew_dict = training_batch.rew_dict
    info_dict = training_batch.info_dict
    bins = training_batch.bins
    scale = training_batch.scale

    gamma, lmbda, eps_clip = hyperparams.gamma, hyperparams.lmbda, hyperparams.eps_clip
    policy_loss_coef = hyperparams.policy_loss_coef
    value_loss_coef = hyperparams.value_loss_coef
    entropy_coef = hyperparams.entropy_coef
    reward_param = hyperparams.reward_param
    mine_feats_names = hyperparams.mine_feats_names

    behav_log_probs = cal_hier_log_probs(act_dict)
    rew_sca = rew_vec_to_scaled_scalar(rew_dict, reward_param)

    is_fir = info_dict["is_fir"]
    hx, cx = obs_dict["hx"], obs_dict["cx"]

    # Model forward pass
    log_probs, entropy, value = jax_model_forward(
        model, obs_dict, act_dict, hx[:, 0], cx[:, 0]
    )

    v_res = TwohotDecoding(value, bins)

    td_target_res = (
        rew_sca[:, :-1]
        + gamma * (1 - is_fir[:, 1:]) * v_res[:, 1:]
    )
    delta = td_target_res - v_res[:, :-1]

    td_target_twohot = TwohotEncoding(td_target_res, bins)
    td_target_twohot = jax.lax.stop_gradient(td_target_twohot) # stop-gradient

    # GAE and normalization
    gae = compute_gae(delta, gamma, lmbda)
    scale = CalculateScale(gae, scale, ALPHA, Q_HIGH, Q_LOW)

    gae = NormReturns(gae, scale)
    gae = jax.lax.stop_gradient(gae) # stop-gradient

    valid_mine_mask = goal_curr_alive_mine_mask(obs_dict["obs_mine"], tuple(mine_feats_names))

    # PPO loss calculations
    ratio = jnp.exp(log_probs[:, :-1] - behav_log_probs[:, :-1])

    # Policy loss 계산
    masked_surr1 = jnp.where(valid_mine_mask, ratio * gae, 0.0)
    masked_surr2 = jnp.where(valid_mine_mask, jax.lax.clamp(1.0 - eps_clip, ratio, 1.0 + eps_clip) * gae, 0.0)

    loss_policy = jax.lax.cond(
        valid_mine_mask.sum() > 0,
        lambda: -jnp.minimum(masked_surr1, masked_surr2).sum() / valid_mine_mask.sum(),
        lambda: jnp.nan,
    )

    # Value loss 계산
    masked_td_loss = cross_entropy_loss(value[:, :-1], td_target_twohot)
    masked_td_loss = jnp.where(valid_mine_mask.squeeze(-1), masked_td_loss, 0.0)

    loss_value = jax.lax.cond(
        valid_mine_mask.sum() > 0,
        lambda: masked_td_loss.sum() / valid_mine_mask.squeeze(-1).sum(),
        lambda: jnp.nan,
    )

    # Policy entropy 계산
    masked_entropy = jnp.where(valid_mine_mask, entropy[:, :-1], 0.0)

    policy_entropy = jax.lax.cond(
        valid_mine_mask.sum() > 0,
        lambda: masked_entropy.sum() / valid_mine_mask.sum(),
        lambda: jnp.nan,
    )

    loss = append_loss(policy_loss_coef * loss_policy)
    loss = append_loss(value_loss_coef * loss_value, loss)
    loss = append_loss(-entropy_coef * policy_entropy, loss)

    return loss, (loss_policy, loss_value, policy_entropy, scale, ratio)


@partial(nnx.jit, static_argnames=["hyperparams"])
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    hyperparams: HyperParams,
    training_batch: TrainingBatch,
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(model, hyperparams, training_batch)
    loss_policy, loss_value, policy_entropy, scale, ratio = aux

    metrics.update(
        total_loss=loss,
        policy_loss=loss_policy,
        value_loss=loss_value,
        policy_entropy=policy_entropy,
        scale=scale,
        min_ratio=ratio.min(),
        max_ratio=ratio.max(),
        avg_ratio=ratio.mean(),
    ) # In-place updates

    optimizer.update(grads)  # In-place updates
    return scale