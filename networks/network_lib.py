import jax
import distrax
import jax.numpy as jnp

from typing import Tuple

from agents.learner_module.learner_lib import Normalizier


Symlog = Normalizier.symlog


def remove_keys_from_state(state, keys_to_remove):
    """
    Recursively removes specified keys from a nested dictionary structure.

    Args:
        state (dict): The nested dictionary structure (e.g., model state).
        keys_to_remove (set): A set of keys to remove from the dictionary.

    Returns:
        dict: The updated dictionary with specified keys removed.
    """
    
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            if k not in keys_to_remove:
                new_state[k] = remove_keys_from_state(v, keys_to_remove)
        return new_state
    elif isinstance(state, (list, tuple)):
        # If state contains lists or tuples, apply removal recursively
        return type(state)(remove_keys_from_state(x, keys_to_remove) for x in state)
    else:
        return state


def reset_rngs_in_state(src_state_pure_dict, dst_state_pure_dict):
    """
    Resets or adds rng keys to the dst_state_pure_dict based on the src_state_pure_dict.
    """
    
    for key, value in src_state_pure_dict.items():
        if key == "rngs":
            # Add or reset rngs based on src_state_pure_dict
            dst_state_pure_dict[key] = {
                subkey: {
                    "count": jnp.array(0, dtype=jnp.uint32),  # Default count
                    "key": jax.random.PRNGKey(0),  # Default key
                }
                for subkey in value  # Keep subkeys like 'default', 'noise', etc.
            }
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            if key not in dst_state_pure_dict:
                dst_state_pure_dict[key] = {}
            reset_rngs_in_state(value, dst_state_pure_dict[key])
        else:
            # Directly copy non-dict values
            dst_state_pure_dict[key] = dst_state_pure_dict.get(key, value)
    return dst_state_pure_dict


@jax.jit
def build_dist(logit: jnp.ndarray, avail: jnp.ndarray):
    """Mask logits based on availability and calculate probabilities."""
    
    # assert logit.shape == avail.shape
    avail = jnp.where(avail == 0, jnp.array(-1e10, dtype=logit.dtype, device=logit.device), jnp.array(1.0, dtype=logit.dtype, device=logit.device))
    masked_logit = logit + avail
    # probs = jax.nn.softmax(masked_logit, axis=-1)
    return distrax.Categorical(logits=masked_logit).sample()


# @jax.jit
# def categorical_sample(rng: PRNGKey, logits: jnp.ndarray) -> jnp.ndarray:
#     """Sample actions from categorical distribution."""
    
#     return jax.random.categorical(rng, logits=logits)


@jax.jit
def get_act_outs(
    logit_act: jnp.ndarray,
    logit_move: jnp.ndarray,
    logit_target: jnp.ndarray,
    avail_tuple: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    seed: jnp.ndarray,  # Random seed
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    avail_act, avail_move, avail_target = avail_tuple

    # Create distributions
    dist_act = build_dist(logit_act, avail_act)
    dist_move = build_dist(logit_move, avail_move)
    dist_target = build_dist(logit_target, avail_target)

    # Split seed for independent sampling
    key1, key2, key3 = jax.random.split(seed, 3)

    # Sample actions
    #TODO: Seed 상태가 ...
    act_sampled = dist_act.sample(seed=key1)
    move_sampled = dist_move.sample(seed=key2)
    target_sampled = dist_target.sample(seed=key3)

    return (
        act_sampled,
        move_sampled,
        target_sampled,
        dist_act.logits,
        dist_move.logits,
        dist_target.logits,
    )


@jax.jit
def get_forward_outs(
    logit_act: jnp.ndarray,
    logit_move: jnp.ndarray,
    logit_target: jnp.ndarray,
    avail_tuple: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    on_select_act: jnp.ndarray,
    on_select_move: jnp.ndarray,
    on_select_target: jnp.ndarray,
    act_sampled: jnp.ndarray,
    move_sampled: jnp.ndarray,
    target_sampled: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    avail_act, avail_move, avail_target = avail_tuple

    # Create distributions
    dist_act = build_dist(logit_act, avail_act)
    dist_move = build_dist(logit_move, avail_move)
    dist_target = build_dist(logit_target, avail_target)

    log_probs = jnp.stack([
        on_select * jnp.expand_dims(dist.log_prob(sampled.squeeze(-1)), -1)
        for on_select, dist, sampled in zip(
            [on_select_act, on_select_move, on_select_target],
            [dist_act, dist_move, dist_target],
            [act_sampled, move_sampled, target_sampled],
            )
        ],
        axis=-1,
    ).sum(-1)

    entropy = jnp.stack([
        on_select * jnp.expand_dims(dist.entropy(), -1)
        for on_select, dist in zip(
            [on_select_act, on_select_move, on_select_target],
            [dist_act, dist_move, dist_target],
            )
        ],
        axis=-1,
    ).sum(-1)

    return log_probs, entropy