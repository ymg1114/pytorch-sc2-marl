import jax
import jax.numpy as jnp

from collections import defaultdict


@jax.jit
def collect_data(values):
    """
    Stack the values into a JAX array.

    Args:
        values (list): List of values for a single key.

    Returns:
        jnp.ndarray: Stacked JAX array.
    """
    return jnp.stack([jnp.array(v) for v in values], axis=0)


def make_as_array_jax(trajectory_obj):
    """
    Optimized version of make_as_array using JAX vmap.

    Args:
        trajectory_obj (Trajectory2): An instance of the Trajectory2 class.

    Returns:
        dict: Dictionary with keys as data fields and values as JAX arrays.
    """
    
    assert trajectory_obj.len > 0

    # Extract keys dynamically from the first rollout (excluding "id")
    keys = [key for key in trajectory_obj.data[0].keys() if key != "id"]

    # Convert trajectory_obj.data to a structured NumPy array
    data_list = trajectory_obj.data  # List of dictionaries

    # Ensure data_list is compatible with JAX operations
    # Convert list of dictionaries into a dictionary of lists
    structured_data = {key: [rollout[key] for rollout in data_list] for key in keys}
    
    # Use vmap to process all keys
    refrased_rollout_data = {key: collect_data(structured_data[key]) for key in keys}

    return refrased_rollout_data


# def make_as_array_origin(trajectory_obj):
#     assert trajectory_obj.len > 0

#     refrased_rollout_data = defaultdict(list)

#     for rollout in trajectory_obj.data:
#         for key, value in rollout.items():
#             if key != "id":  # 학습 데이터만 취급
#                 refrased_rollout_data[key].append(value)

#     refrased_rollout_data = {
#         k: jnp.stack(v, 0) for k, v in refrased_rollout_data.items()
#     }
#     return refrased_rollout_data


def rearrange_data_origin(data):
    arranged_data = defaultdict(dict)
    
    ids = data["id"].tolist()  # 각 경기 별 고유 에이전트 id
    obs_dict = data["obs_dict"]
    act_dict = data["act_dict"]
    rew_vec = data["rew_vec"]
    is_fir = data["is_fir"]
    done = data["done"]
    
    for idx, a_id in enumerate(ids):
        arranged_data[a_id].update({k: v[idx] for k, v in obs_dict.items()})  # (dim, )
        arranged_data[a_id].update({k: v[idx] for k, v in act_dict.items()})  # (dim, )

        arranged_data[a_id]["rew_vec"] = rew_vec[idx] # (rev_d, )
        arranged_data[a_id]["is_fir"] = jnp.expand_dims(is_fir[idx], axis=-1) # (1, )
        arranged_data[a_id]["done"] = jnp.expand_dims(done[idx], axis=-1) # (1, )
    
    return arranged_data
