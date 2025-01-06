import os
import time
import asyncio
# import numpy as np

from typing import NamedTuple, Optional, Dict, List, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import distrax

from flax import nnx
from flax.core import FrozenDict

from utils.utils import ExecutionTimer

from rewarder.rewarder import REWARD_PARAM


# Normalizing 's'
ALPHA = 0.99
Q_HIGH = 0.95
Q_LOW = 0.05


# IMPALA HyperParams
RHO_BAR = 0.8
C_BAR = 1.0


class HyperParams(NamedTuple):
    gamma: float
    lmbda: float
    eps_clip: float
    policy_loss_coef: float
    value_loss_coef: float
    entropy_coef: float
    reward_param: Tuple[Tuple[str, float], ...]
    mine_feats_names: Tuple[str, ...]


class TrainingBatch(NamedTuple):
    obs_dict: FrozenDict[str, jax.Array]
    act_dict: FrozenDict[str, jax.Array]
    rew_dict: FrozenDict[str, jax.Array]
    info_dict: FrozenDict[str, jax.Array]
    bins: jax.Array
    scale: jax.Array


@partial(jax.jit, static_argnames=["device"])
def move_to_device(batch_dict, device):
    return FrozenDict(jax.tree.map(lambda v: jax.device_put(v, device), batch_dict))


@partial(jax.jit, static_argnames=["device"])
def jax_device_movement(batch_dict, device):
    obs_dict = move_to_device(batch_dict["obs"], device)
    act_dict = move_to_device(batch_dict["act"], device)
    rew_dict = move_to_device(batch_dict["rew"], device)
    info_dict = move_to_device(batch_dict["info"], device)
    return obs_dict, act_dict, rew_dict, info_dict


@jax.jit
def append_loss(trg_loss, src_loss=jnp.nan):
    return jax.lax.cond(
        jnp.isnan(src_loss).any(),
        lambda _: trg_loss,
        lambda _: jax.lax.cond(
            jnp.isnan(trg_loss).any(),
            lambda _: src_loss,
            lambda _: src_loss + trg_loss,
            operand=None
        ),
        operand=None
    )


@partial(jax.jit, static_argnums=(1,))
@partial(jax.vmap, in_axes=(0, None), out_axes=0)
def goal_curr_alive_mine_mask(obs_mine, mine_feats_names):
    # Extract observations
    # curr_obs_mine = obs_mine[:, :-1] # current-obs 개념 / Considering Batch-Dim
    curr_obs_mine = obs_mine[:-1, ...]  # current-obs 개념

    # current 기준, 살아있는 나 (mine) 여부 확인, Shape: [Seq, Dim]
    curr_mine_health = curr_obs_mine[..., mine_feats_names.index('own_health')]
    valid_mine_mask = curr_mine_health > 0  # current 기준, 살아있는 나 (mine) 여부 확인

    # Expand dimensions for compatibility (add trailing dim for [Seq, 1])
    valid_mine_mask = jnp.expand_dims(valid_mine_mask, axis=-1)

    return valid_mine_mask


@partial(jax.jit, donate_argnames="deltas")
@partial(jax.vmap, in_axes=(0, None, None), out_axes=0)
def compute_gae(
    deltas,
    gamma,
    lambda_,
):
    """
    Compute Generalized Advantage Estimation (GAE) using jax.lax.scan.

    Args:
        deltas (jnp.ndarray): TD-errors, shape (Batch, Sequence, Dim).
        gamma (float): Discount factor.
        lambda_ (float): GAE parameter.

    Returns:
        jnp.ndarray: GAE returns, shape (Batch, Sequence, Dim).
    """

    # Define the reverse scanning function
    def scan_fn(carry, delta):
        gae = delta + gamma * lambda_ * carry
        return gae, gae
    
    # Reverse deltas along the sequence dimension
    reversed_deltas = deltas[::-1]
    
    # Ensure init_carry matches the shape of deltas' last dimension
    init_carry = jnp.zeros(deltas.shape[-1])  # Shape: (Dim,)
    
    # Perform scan to compute GAE
    _, reversed_returns = jax.lax.scan(scan_fn, init_carry, reversed_deltas)

    # Reverse the result to restore original sequence order
    returns = reversed_returns[::-1]
    return returns


@partial(jax.jit, donate_argnames=["behav_log_probs", "is_fir", "rewards"])
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None, None, None), out_axes=(0, 0, 0))
def compute_v_trace(
    behav_log_probs,
    target_log_probs,
    is_fir,
    rewards,
    values,
    gamma,
    rho_bar,
    c_bar,
):
    # Importance sampling weights (rho)
    rho = jnp.exp(
        target_log_probs[:-1, ...] - behav_log_probs[:-1, ...]
    )
    rho_clipped = jnp.clip(rho, min=0.1, max=rho_bar)

    # Truncated importance weights (c)
    c = jnp.exp(
        target_log_probs[:-1, ...] - behav_log_probs[:-1, ...]
    )
    c_clipped = jnp.clip(c, max=c_bar)

    # Considering Batch-Dim
    td_target = rewards[:-1, ...] + gamma * (1 - is_fir[1:, ...]) * values[1:, ...]
    deltas = rho_clipped * (td_target - values[:-1, ...])  # TD-Error with clipping 보정

    def scan_fn(carry, t):
        vs_minus_v_xs_next = carry
        vs_minus_v_xs = (
            deltas[t, ...]
            + c_clipped[t, ...]
            * (gamma * (1 - is_fir[t + 1, ...]) * vs_minus_v_xs_next[t + 1, ...])
        )
        updated_carry = carry.at[t, ...].set(vs_minus_v_xs)
        return updated_carry, None

    init_carry = jnp.zeros_like(values)
    vs_minus_v_xs, _ = jax.lax.scan(scan_fn, init_carry, jnp.arange(deltas.shape[0])[::-1])

    # vs_minus_v_xs를 V-trace를 통해 수정된 가치 추정치
    values_target = values + vs_minus_v_xs

    advantages = rho_clipped * (
        rewards[:-1, ...]
        + gamma * (1 - is_fir[1:, ...]) * values_target[1:, ...]
        - values[:-1, ...]
    )

    return rho_clipped, advantages, values_target


@partial(jax.jit, donate_argnames=["behav_log_probs", "is_fir", "rewards"])
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None, None, None), out_axes=(0, 0, 0))
def compute_v_trace_twohot(
    behav_log_probs,
    target_log_probs,
    is_fir,
    rewards,
    v_res,
    gamma,
    rho_bar,
    c_bar,
):
    # Importance sampling weights (rho)
    rho = jnp.exp(
        target_log_probs[:-1, ...] - behav_log_probs[:-1, ...]
    )
    rho_clipped = jnp.clip(rho, min=0.1, max=rho_bar)

    # Truncated importance weights (c)
    c = jnp.exp(
        target_log_probs[:-1, ...] - behav_log_probs[:-1, ...]
    )
    c_clipped = jnp.clip(c, max=c_bar)

    td_target_res = rewards[:-1, ...] + gamma * (1 - is_fir[1:, ...]) * v_res[1:, ...]
    deltas = rho_clipped * (td_target_res - v_res[:-1, ...])  # TD-Error with clipping 보정

    def scan_fn(carry, t):
        vs_minus_v_xs_next = carry
        vs_minus_v_xs = (
            deltas[t, ...]
            + c_clipped[t, ...]
            * (gamma * (1 - is_fir[t + 1, ...]) * vs_minus_v_xs_next[t + 1, ...])
        )
        updated_carry = carry.at[t, ...].set(vs_minus_v_xs)
        return updated_carry, None

    init_carry = jnp.zeros_like(v_res)
    vs_minus_v_xs, _ = jax.lax.scan(scan_fn, init_carry, jnp.arange(deltas.shape[0])[::-1])

    # vs_minus_v_xs는 V-trace를 통해 수정된 가치 추정치
    values_target = v_res + vs_minus_v_xs

    advantages = rho_clipped * (
        rewards[:-1, ...]
        + gamma * (1 - is_fir[1:, ...]) * values_target[1:, ...]
        - v_res[:-1, ...]
    )

    return rho_clipped, advantages, values_target


@jax.jit
@partial(jax.vmap, in_axes=(0, 0), out_axes=0)
def kldivergence(logits_p, logits_q):
    """
    Compute KL divergence between two categorical distributions.
    """

    log_probs_p = jax.nn.log_softmax(logits_p, axis=-1)
    log_probs_q = jax.nn.log_softmax(logits_q, axis=-1)
    
    probs_p = jax.nn.softmax(logits_p, axis=-1)
    return jnp.sum(probs_p * (log_probs_p - log_probs_q), axis=-1)


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, 0), out_axes=0)
def cal_log_probs(logit, sampled, on_select):
    """
    Calculate log probabilities for sampled values.

    Args:
        logit: Logits for the categorical distribution.
        sampled: Sampled values.
        on_select: Selector mask for the log probabilities.

    Returns:
        Log probabilities of the sampled values.
    """
    
    # dist = distrax.Categorical(probs=jax.nn.softmax(logit, axis=-1))
    dist = distrax.Categorical(logits=logit)
    log_probs = dist.log_prob(sampled.squeeze(-1))
    return on_select * log_probs[..., jnp.newaxis]


@jax.jit
@partial(jax.vmap, in_axes=(0, 0), out_axes=0)
def cross_entropy_loss(logits, targets):
    """
    Compute the cross-entropy loss.

    Args:
        logits: Logits for the predictions.
        targets: One-hot encoded target probabilities.

    Returns:
        Cross-entropy loss.
    """
    
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(targets * log_probs, axis=-1)


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


@partial(jax.jit, static_argnums=1)
def rew_vec_to_scaled_scalar(rew_dict, reward_param):
    rew_vec = rew_dict["rew_vec"]

    # B, S, D = rew_vec.shape
    # assert D == len(reward_param)

    weights = jnp.array([value for _, value in reward_param], dtype=rew_vec.dtype)
    scaled_rew_vec = rew_vec * weights

    return jnp.sum(scaled_rew_vec, axis=-1, keepdims=True)


class Normalizier():
    """참고, DreamerV3: https://arxiv.org/pdf/2301.04104
    """
    
    @staticmethod
    @jax.jit
    def symlog(x):
        return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)

    @staticmethod
    @jax.jit
    def symexp(x):
        return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

    @staticmethod
    @jax.jit
    def twohot_decoding(logits, bins):
        probs = jax.nn.softmax(logits, axis=-1)
        bin_positions = bins
        return jnp.sum(probs * bin_positions, axis=-1, keepdims=True)

    @staticmethod
    @jax.jit
    def twohot_encoding(scalars, bins):
        """
        Twohot encodes a batch of scalars, ensuring that two values sum to 1.

        Args:
            scalars (jnp.ndarray): A tensor of shape (Batch, Sequence, 1) containing scalar values.
            bins (jnp.ndarray): A tensor of shape (N,) representing discrete bin positions.

        Returns:
            jnp.ndarray: A tensor of shape (Batch, Sequence, N) containing twohot-encoded vectors
                        where two adjacent bins have non-zero weights summing to 1.
        """
        
        # Convert bins to JAX array
        bins = jnp.asarray(bins)

        # Expand scalars and bins for broadcasting
        scalars_expanded = jnp.broadcast_to(scalars, (*scalars.shape[:-1], bins.size))  # Shape: (Batch, Sequence, N)
        bins_expanded = jnp.broadcast_to(
            jnp.expand_dims(jnp.expand_dims(bins, 0), 0),  # Add two singleton dimensions
            (scalars.shape[0], scalars.shape[1], bins.shape[0])  # Target shape: (Batch, Sequence, N)
        )

        # Compute the absolute differences between scalars and bins
        diffs = jnp.abs(bins_expanded - scalars_expanded)

        # Identify the index of the closest bin (lower bin)
        lower_bin = jnp.argmin(diffs, axis=-1, keepdims=True)  # Shape: (Batch, Sequence, 1)
        lower_bin_value = jnp.take_along_axis(bins_expanded, lower_bin, axis=-1)  # Bin value at the lower index

        # Determine the upper bin based on the scalar's position
        upper_bin = jnp.where(
            scalars >= lower_bin_value,
            lower_bin + 1,  # Upper bin is next to the lower bin
            lower_bin - 1   # Upper bin is the previous bin
        )

        # Clip indices to ensure they are within the valid range of bins
        upper_bin = jnp.clip(upper_bin, 0, bins.size - 1)
        lower_bin = jnp.clip(lower_bin, 0, bins.size - 1)

        # Retrieve the values of the upper and lower bins
        upper_bin_value = jnp.take_along_axis(bins_expanded, upper_bin, axis=-1)
        lower_bin_value = jnp.take_along_axis(bins_expanded, lower_bin, axis=-1)

        # Calculate weights for the lower and upper bins
        # Avoid division by zero by adding a small epsilon to the denominator
        denom = upper_bin_value - lower_bin_value + 1e-10
        lower_weight = (upper_bin_value - scalars) / denom
        upper_weight = 1.0 - lower_weight

        # Initialize the twohot vector with zeros
        twohot_vector = jnp.zeros_like(scalars_expanded)  # Shape: (Batch, Sequence, N)
        
        # Create batch and sequence indices for scatter_add operations
        batch_indices = jnp.arange(twohot_vector.shape[0])[:, None, None]  # Shape: (Batch, 1, 1)
        seq_indices = jnp.arange(twohot_vector.shape[1])[None, :, None]    # Shape: (1, Sequence, 1)

        # Broadcast indices to match the shape of the lower_bin
        batch_indices = jnp.broadcast_to(batch_indices, lower_bin.shape)  # Shape: (Batch, Sequence, 1)
        seq_indices = jnp.broadcast_to(seq_indices, lower_bin.shape)      # Shape: (Batch, Sequence, 1)

        # Flatten all indices and weights for 1D scatter_add operation
        flat_batch_indices = batch_indices.flatten()
        flat_seq_indices = seq_indices.flatten()
        flat_lower_bin = lower_bin.squeeze(-1).flatten()
        flat_upper_bin = upper_bin.squeeze(-1).flatten()
        flat_lower_weight = lower_weight.squeeze(-1).flatten()
        flat_upper_weight = upper_weight.squeeze(-1).flatten()

        # Flatten the twohot_vector to prepare for scatter operations
        flat_twohot_vector = twohot_vector.reshape(-1, twohot_vector.shape[-1])

        # Scatter lower weights into the twohot vector
        flat_twohot_vector = flat_twohot_vector.at[
            (flat_batch_indices * twohot_vector.shape[1] + flat_seq_indices, flat_lower_bin)
        ].add(flat_lower_weight)

        # Scatter upper weights into the twohot vector
        flat_twohot_vector = flat_twohot_vector.at[
            (flat_batch_indices * twohot_vector.shape[1] + flat_seq_indices, flat_upper_bin)
        ].add(flat_upper_weight)

        # Reshape the flattened twohot_vector back to its original shape
        twohot_vector = flat_twohot_vector.reshape(twohot_vector.shape)

        return twohot_vector

    @staticmethod
    @jax.jit
    def norm_returns(returns, s):
        return returns / jnp.maximum(1, s)
    
    @staticmethod
    @partial(jax.jit, static_argnums=(2,3,4))
    def calculate_s(
        returns: jnp.ndarray,
        previous_s: jnp.ndarray,
        alpha: float = 0.99,
        q_high: float = 0.95,
        q_low: float = 0.05
    ) -> jnp.ndarray:
        """
        Calculate the normalization factor "s" using the q_low th and q_high th percentile of returns.
        This function applies an exponential moving average (EMA) to smooth the value of s.
        
        Args:
            returns (jnp.ndarray): Tensor of shape (Batch, Sequence, 1) containing the return estimates.
            previous_s (jnp.ndarray): Previous value of s for EMA. If None, this is the first iteration.
            alpha (float): Smoothing factor for the exponential moving average (EMA decay).
            q_high (float): Upper quantile for normalization (default is 0.95).
            q_low (float): Lower quantile for normalization (default is 0.05).
            
        Returns:
            jnp.ndarray: The normalization factor s (a scalar tensor).
        """
        
        # Remove the last dimension to calculate percentiles across the Batch
        returns = returns.squeeze(-1)  # Shape: (Batch, Sequence)

        # Calculate the specified upper and lower percentiles along the batch dimension
        upper = jnp.quantile(returns, q_high, axis=0)  # Shape: (Sequence,)
        lower = jnp.quantile(returns, q_low, axis=0)   # Shape: (Sequence,)

        # Compute the difference between the percentiles
        diff = upper - lower  # Shape: (Sequence,)

        # Calculate the current s value as the mean of the differences across the sequence
        s_current = jnp.mean(diff)  # Scalar value

        # Condition function
        def if_nan(previous_s, s_current):
            # Case where previous_s is nan (None equivalent)
            return s_current

        def if_not_nan(previous_s, s_current):
            # Apply EMA logic when previous_s is not None
            return (1 - alpha) * previous_s + alpha * s_current

        # Use lax.cond to handle the conditional logic
        return jax.lax.cond(
            jnp.isnan(previous_s),  # Check if previous_s is nan
            if_nan,
            if_not_nan,
            previous_s,
            s_current,
        )


async def learning(parent, train_step, timer: ExecutionTimer):
    assert hasattr(parent, "batch_queue")
    scale = parent.scale  # 초기화

    while not parent.stop_event.is_set():
        batch_dict = None
        with timer.timer("learner-throughput", check_throughput=True):
            with timer.timer("learner-batching-time"):
                batch_dict = await parent.batch_queue.get()

        if batch_dict is not None:
            with timer.timer("learner-forward-time"):
                # Basically, mini-batch-learning (batch, seq, feat)
                assert "obs" in batch_dict
                assert "act" in batch_dict
                assert "rew" in batch_dict
                assert "info" in batch_dict

                # Move data to the appropriate device
                # obs_dict = {k: jax.device_put(v, parent.device) for k, v, in batch_dict["obs"].items()}
                # act_dict = {k: jax.device_put(v, parent.device) for k, v, in batch_dict["act"].items()}
                # rew_dict = {k: jax.device_put(v, parent.device) for k, v, in batch_dict["rew"].items()}
                # info_dict = {k: jax.device_put(v, parent.device) for k, v, in batch_dict["info"].items()}
                obs_dict, act_dict, rew_dict, info_dict = jax_device_movement(
                    batch_dict, parent.device
                )

                training_batch = TrainingBatch(
                    obs_dict=obs_dict,
                    act_dict=act_dict,
                    rew_dict=rew_dict,
                    info_dict=info_dict,
                    bins=parent.model.bins,
                    scale=scale,
                )

                hyperparams = HyperParams(
                    gamma=parent.args.gamma,
                    lmbda=parent.args.lmbda,
                    eps_clip=parent.args.eps_clip,
                    policy_loss_coef=parent.args.policy_loss_coef,
                    value_loss_coef=parent.args.value_loss_coef,
                    entropy_coef=parent.args.entropy_coef,
                    reward_param=tuple(REWARD_PARAM.items()),
                    mine_feats_names=tuple(parent.env_space["others"]["mine_feats_names"]),
                )

                # epoch-learning
                for _ in range(parent.args.K_epoch):
                    # 노말라이징 scale을 지속적으로 업데이트
                    scale = train_step(
                        parent.model,
                        parent.optimizer,
                        parent.metrics,
                        hyperparams,
                        training_batch,
                    )

                    with timer.timer("learner-backward-time"):
                        print(
                            "loss: {:.5f} original_value_loss: {:.5f} original_policy_loss: {:.5f} "
                            "original_policy_entropy: {:.5f} ratio-avg: {:.5f}".format(
                                parent.metrics.total_loss.compute(),
                                parent.metrics.value_loss.compute(),
                                parent.metrics.policy_loss.compute(),
                                parent.metrics.policy_entropy.compute(),
                                parent.metrics.avg_ratio.compute(),
                            )
                        )

                parent.pub_model(nnx.state(parent.model).to_pure_dict())

                if parent.idx % parent.args.loss_log_interval == 0:
                    await parent.log_loss_tensorboard(timer)

                if parent.idx % parent.args.model_save_interval == 0:
                    parent.model.save_model_weight(
                        os.path.join(parent.args.model_dir, f"{parent.args.algo}_{parent.idx}.pt"),
                        parent.idx,
                        scale,
                        parent.model,
                        nnx.state(parent.optimizer),
                    )

                parent.idx += 1

            if parent.heartbeat is not None:
                parent.heartbeat.value = time.monotonic()

        await asyncio.sleep(1e-4)
