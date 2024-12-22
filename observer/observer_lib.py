import functools

import jax.numpy as jnp
from jax import lax, jit, vmap


# @jit
# def _jit_update_hp(current_hp, hp_delta, last_hp):
#     # 첫 번째 스텝에서는 변화량을 0으로 설정
#     def compute_zero_delta(_):
#         return jnp.zeros_like(current_hp)

#     # 이후 스텝에서는 정상적으로 변화량 계산
#     def compute_delta(_):
#         return current_hp - last_hp

#     # Use lax.cond for the if/else logic
#     delta = lax.cond(jnp.all(last_hp == 0), compute_zero_delta, compute_delta, operand=None)

#     # Update hp_delta with the computed delta
#     hp_delta = jnp.concatenate([hp_delta[..., 1:], delta[..., jnp.newaxis]], axis=-1)
#     last_hp = current_hp
#     return hp_delta, last_hp


@jit
@functools.partial(vmap, in_axes=(0, 0, 0, None), out_axes=(0, 0))
def _jit_update_hp_vmap(current_hp, hp_delta, last_hp, last_hp_for_check):
    # 첫 번째 스텝에서는 변화량을 0으로 설정
    def compute_zero_delta(_):
        return jnp.zeros_like(current_hp)

    # 이후 스텝에서는 정상적으로 변화량 계산
    def compute_delta(_):
        return current_hp - last_hp

    # Use lax.cond for the if/else logic
    delta = lax.cond(jnp.all(last_hp_for_check == 0), compute_zero_delta, compute_delta, operand=None)

    # Update hp_delta with the computed delta
    hp_delta = jnp.concatenate([hp_delta[..., 1:], delta[..., jnp.newaxis]], axis=-1)
    last_hp = current_hp
    return hp_delta, last_hp


# @functools.partial(jit, static_argnums=(4,))
# def _jit_get_obs(obs_mine, obs_ally, obs_enemy, hp_delta, n_allies):
#     mean_hp_delta = hp_delta.mean(-1)

#     # 나 (mine)의 feature에 mean_hp_delta 추가
#     mine_hp_delta = mean_hp_delta[:, 0:1] # (n_agents, 1)
#     obs_mine = jnp.concatenate([obs_mine, mine_hp_delta], axis=-1)

#     # 아군 (ally)의 feature에 mean_hp_delta 추가
#     ally_hp_delta = mean_hp_delta[:, 1 : n_allies + 1] # (n_agents, n_allies)
#     obs_ally = jnp.concatenate([obs_ally, ally_hp_delta], axis=-1)

#     # 적군 (enemy)의 feature에 mean_hp_delta 추가
#     enemy_hp_delta = mean_hp_delta[:, n_allies + 1 :] # (n_agents, n_enemies)
#     obs_enemy = jnp.concatenate([obs_enemy, enemy_hp_delta], axis=-1)

#     return obs_mine, obs_ally, obs_enemy


@functools.partial(jit, static_argnums=(4,))
@functools.partial(vmap, in_axes=(0, 0, 0, 0, None), out_axes=(0, 0, 0))
def _jit_get_obs_vmap(obs_mine, obs_ally, obs_enemy, hp_delta, n_allies):
    mean_hp_delta = hp_delta.mean(-1)

    # Add mean_hp_delta to mine features
    mine_hp_delta = mean_hp_delta[..., 0:1]  # (n_agents, 1)
    obs_mine = jnp.concatenate([obs_mine, mine_hp_delta], axis=-1)

    # Add mean_hp_delta to ally features
    ally_hp_delta = mean_hp_delta[..., 1 : n_allies + 1]  # (n_agents, n_allies)
    obs_ally = jnp.concatenate([obs_ally, ally_hp_delta], axis=-1)

    # Add mean_hp_delta to enemy features
    enemy_hp_delta = mean_hp_delta[..., n_allies + 1 :]  # (n_agents, n_enemies)
    obs_enemy = jnp.concatenate([obs_enemy, enemy_hp_delta], axis=-1)

    # Return arrays instead of a dict for JAX compatibility
    return obs_mine, obs_ally, obs_enemy
