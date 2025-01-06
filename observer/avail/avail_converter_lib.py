import jax
import jax.numpy as jnp

from functools import partial


@partial(jax.jit, static_argnames=("n_agents", "dim_act", "dim_move", "no_op_idx", "stop_idx", "move_idx", "target_idx", "flee_idx"))
def avail_act_parsing_jit(avail_total_act, valid_flee_indices, n_agents, dim_act, dim_move, no_op_idx, stop_idx, move_idx, target_idx, flee_idx):
    """
    JAX JIT 컴파일 가능한 avail_act_parsing 함수
    """
    
    # 초기화
    avail_act = jnp.zeros((n_agents, dim_act), dtype=jnp.float32) # (n_ally, dim_act)

    # 업데이트
    avail_act = avail_act.at[:, no_op_idx].set(avail_total_act[:, no_op_idx])  # no-op
    avail_act = avail_act.at[:, stop_idx].set(avail_total_act[:, stop_idx])  # stop
    avail_act = avail_act.at[:, move_idx].set(avail_total_act[:, move_idx: move_idx+dim_move].any(-1))  # move
    avail_act = avail_act.at[:, target_idx].set(avail_total_act[:, move_idx+dim_move:-1].any(-1))  # target + Flee action

    # Flee action
    avail_act = avail_act.at[valid_flee_indices, flee_idx].set(1.0)

    return avail_act


@partial(jax.jit, static_argnames=("n_agents", "dim_move", "move_north_idx", "move_south_idx", "move_east_idx", "move_west_idx"))
def avail_move_parsing_jit(avail_total_act, n_agents, dim_move, move_north_idx, move_south_idx, move_east_idx, move_west_idx):
    """
    JAX JIT 컴파일 가능한 avail_move_parsing 함수
    """
    
    # 초기화
    avail_move = jnp.zeros((n_agents, dim_move), dtype=jnp.float32)  # (n_ally, dim_move)

    # 이동 방향별 업데이트
    avail_move = avail_move.at[:, 0].set(avail_total_act[:, move_north_idx])  # north
    avail_move = avail_move.at[:, 1].set(avail_total_act[:, move_south_idx])  # south
    avail_move = avail_move.at[:, 2].set(avail_total_act[:, move_east_idx])   # east
    avail_move = avail_move.at[:, 3].set(avail_total_act[:, move_west_idx])   # west

    return avail_move


@partial(jax.jit, static_argnames=("n_agents", "dim_target", "move_idx", "dim_move"))
def avail_target_parsing_jit(avail_total_act, n_agents, dim_target, move_idx, dim_move):
    """
    JAX JIT 컴파일 가능한 avail_target_parsing 함수
    """
    
    # 초기화
    avail_target = jnp.zeros((n_agents, dim_target), dtype=jnp.float32)  # (n_ally, dim_target)

    # 타겟 데이터 슬라이싱
    avail_target = avail_target.at[:, :].set(avail_total_act[:, move_idx+dim_move:-1])  # target + Flee action
    
    return avail_target
