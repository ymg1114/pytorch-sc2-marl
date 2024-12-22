import numpy as np

from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp
import distrax

from rewarder.rewarder import REWARD_PARAM


@jax.jit
def append_loss(trg_loss, src_loss=None):
    return jax.lax.cond(
        src_loss is None,
        lambda _: trg_loss,
        lambda src_loss: jax.lax.cond(
            jnp.isnan(trg_loss).any(),
            lambda _: src_loss,
            lambda _: src_loss + trg_loss,
            operand=None
        ),
        operand=src_loss
    )


@jax.jit
@partial(jax.vmap, in_axes=(0, None), out_axes=0)
def goal_curr_alive_mine_mask(obs_mine, mine_feats_names):
    # Extract observations
    obs_mine = obs_mine[:, 1:]  # next-obs

    curr_obs_mine = obs_mine[:, :-1] # current-obs 개념

    # assert obs_mine.shape[:-1] == obs_ally.shape[:-1] == obs_enemy.shape[:-1]
    
    # current 기준, 살아있는 나 (mine) 여부 확인, Shape: [Batch, Seq]
    curr_mine_health = curr_obs_mine[..., mine_feats_names.index('own_health')]
    valid_mine_mask = curr_mine_health > 0  # current 기준, 살아있는 나 (mine) 여부 확인
    
    # Expand dimensions for compatibility (add trailing dim for [Batch, Seq, 1])
    valid_mine_mask = jnp.expand_dims(valid_mine_mask, axis=-1)
    
    return valid_mine_mask


@jax.jit
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


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None, None, None), out_axes=(0, 0, 0))
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
    rho = jnp.exp(
        target_log_probs[:, :-1] - behav_log_probs[:, :-1]
    )  # a/b == exp(log(a)-log(b))
    # rho_clipped = jnp.clip(rho, max=rho_bar)
    rho_clipped = jnp.clip(rho, min=0.1, max=rho_bar)

    # truncated importance weights (c)
    c = jnp.exp(
        target_log_probs[:, :-1] - behav_log_probs[:, :-1]
    )  # a/b == exp(log(a)-log(b))
    c_clipped = jnp.clip(c, max=c_bar)

    td_target = rewards[:, :-1] + gamma * (1 - is_fir[:, 1:]) * values[:, 1:]
    deltas = rho_clipped * (td_target - values[:, :-1])  # TD-Error with 보정

    def scan_fn(carry, t):
        vs_minus_v_xs_next = carry
        vs_minus_v_xs = (
            deltas[:, t]
            + c_clipped[:, t]
            * (gamma * (1 - is_fir))[:, t + 1]
            * vs_minus_v_xs_next[:, t + 1]
        )
        updated_carry = carry.at[:, t].set(vs_minus_v_xs)
        return updated_carry, vs_minus_v_xs

    init_carry = jnp.zeros_like(values, device=values.device)
    vs_minus_v_xs, _ = jax.lax.scan(scan_fn, init_carry, jnp.arange(deltas.shape[1])[::-1])

    # vs_minus_v_xs는 V-trace를 통해 수정된 가치 추정치
    values_target = values + vs_minus_v_xs

    advantages = rho_clipped * (
        rewards[:, :-1]
        + gamma * (1 - is_fir[:, 1:]) * values_target[:, 1:]
        - values[:, :-1]
    )

    return rho_clipped, advantages, values_target


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None, None, None), out_axes=(0, 0, 0))
def compute_v_trace_twohot(
    behav_log_probs,
    target_log_probs,
    is_fir,
    rewards,
    values,
    gamma,
    twohot_decoding,
    rho_bar=0.8,
    c_bar=1.0,
):
    v_res = twohot_decoding(values)
    
    # Importance sampling weights (rho)
    rho = jnp.exp(
        target_log_probs[:, :-1] - behav_log_probs[:, :-1]
    )  # a/b == exp(log(a)-log(b))
    # rho_clipped = torch.clamp(rho, max=rho_bar)
    rho_clipped = jnp.clip(rho, min=0.1, max=rho_bar)

    # truncated importance weights (c)
    c = jnp.exp(
        target_log_probs[:, :-1] - behav_log_probs[:, :-1]
    )  # a/b == exp(log(a)-log(b))
    c_clipped = jnp.clip(c, max=c_bar)
    
    td_target_res = rewards[:, :-1] + gamma * (1 - is_fir[:, 1:]) * v_res[:, 1:]
    deltas = rho_clipped * (td_target_res - v_res[:, :-1])  # TD-Error with 보정

    B, S, L = values.shape
    
    def scan_fn(carry, t):
        vs_minus_v_xs_next = carry
        vs_minus_v_xs = (
            deltas[:, t]
            + c_clipped[:, t]
            * (gamma * (1 - is_fir))[:, t + 1]
            * vs_minus_v_xs_next[:, t + 1]
        )
        updated_carry = carry.at[:, t].set(vs_minus_v_xs)
        return updated_carry, vs_minus_v_xs

    init_carry = jnp.zeros((B, S, 1), device=values.device)
    vs_minus_v_xs, _ = jax.lax.scan(scan_fn, init_carry, jnp.arange(deltas.shape[1])[::-1])

    # vs_minus_v_xs는 V-trace를 통해 수정된 가치 추정치
    values_target = v_res + vs_minus_v_xs

    advantages = rho_clipped * (
        rewards[:, :-1]
        + gamma * (1 - is_fir[:, 1:]) * values_target[:, 1:]
        - v_res[:, :-1]
    )

    return rho_clipped, advantages, values_target


@jax.jit
def kldivergence(logits_p, logits_q):
    """
    Compute KL divergence between two categorical distributions.
    """

    log_probs_p = jax.nn.log_softmax(logits_p, axis=-1)
    log_probs_q = jax.nn.log_softmax(logits_q, axis=-1)
    
    probs_p = jax.nn.softmax(logits_p, axis=-1)
    return jnp.sum(probs_p * (log_probs_p - log_probs_q), axis=-1)


@jax.jit
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
    log_probs = dist.log_prob(jnp.squeeze(sampled, -1))
    return on_select * log_probs[..., jnp.newaxis]


@jax.jit
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


@jax.jit
def rew_vec_to_scaled_scalar(rew_dict):
    rew_vec = rew_dict["rew_vec"]
    
    # B, S, D = rew_vec.shape
    # assert D == len(REWARD_PARAM)

    weights = jnp.array(list(REWARD_PARAM.values()), dtype=rew_vec.dtype, device=rew_vec.device)
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
    @partial(jax.jit, static_argnums=(2,))
    def calculate_s(
        returns: jnp.ndarray,
        alpha: float = 0.99,
        previous_s: Optional[jnp.ndarray] = None,
        q_high: float = 0.95,
        q_low: float = 0.05
    ) -> jnp.ndarray:
        """
        Calculate the normalization factor "s" using the q_low th and q_high th percentile of returns.
        This function applies an exponential moving average (EMA) to smooth the value of s.
        
        Args:
            returns (torch.Tensor): Tensor of shape (Batch, Sequence, 1) containing the return estimates.
            alpha (float): Smoothing factor for the exponential moving average (EMA decay).
            previous_s (Optional[torch.Tensor]): Previous value of s for EMA. If None, this is the first iteration.
            q_high (float): Upper quantile for normalization (default is 0.95).
            q_low (float): Lower quantile for normalization (default is 0.05).
            
        Returns:
            torch.Tensor: The normalization factor s (a scalar tensor).
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

        # If previous_s is None, this is the first iteration
        if previous_s is None:
            return s_current

        # Apply the modified EMA to favor the current s more
        s = (1 - alpha) * previous_s + alpha * s_current

        return s