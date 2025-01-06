import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from flax import nnx

from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Tuple, Union

if TYPE_CHECKING:
    from env.env_proxy import EnvSpace

from utils.utils import extract_file_num
from .network_lib import Symlog, save_state, load_state, get_act_outs, get_forward_outs


class ModelSingle(nnx.Module):
    def __init__(self, args, env_space: "EnvSpace", rngs=nnx.Rngs(0, noise=1)):
        super().__init__()
        # Network dimension setting
        # self.args = args
        self.hidden_size = args.hidden_size

        obs_space = env_space["obs"]
        act_space = env_space["act"]

        obs_mine_shape = obs_space["obs_mine"].nvec
        obs_ally_shape = obs_space["obs_ally"].nvec
        obs_enemy_shape = obs_space["obs_enemy"].nvec

        logit_act_shape = act_space["logit_act"].nvec
        logit_move_shape = act_space["logit_move"].nvec
        logit_target_shape = act_space["logit_target"].nvec

        # Encode layers
        self.encode_mine = nnx.Linear(int(obs_mine_shape[-1]), self.hidden_size, rngs=rngs)
        self.encode_ally = nnx.Linear(int(obs_ally_shape[-1]), self.hidden_size, rngs=rngs)
        self.encode_enemy = nnx.Linear(int(obs_enemy_shape[-1]), self.hidden_size, rngs=rngs)
        self.encode_body = nnx.Linear(self.hidden_size * 3, self.hidden_size, rngs=rngs)
        self.encode_attn = nnx.Linear(self.hidden_size, self.hidden_size, rngs=rngs)

        # Normalization layers
        self.norm_mine = nnx.LayerNorm(self.hidden_size, rngs=rngs)
        # self.norm_ally = nnx.LayerNorm(self.hidden_size, rngs=rngs)
        # self.norm_enemy = nnx.LayerNorm(self.hidden_size, rngs=rngs)
        self.norm_body = nnx.LayerNorm(self.hidden_size, rngs=rngs)
        self.norm_attn = nnx.LayerNorm(self.hidden_size, rngs=rngs)

        # # Attention
        # self.mine_attn_v = nnx.Linear(int(obs_mine_shape[-1]), self.hidden_size, rngs=rngs)
        # self.multihead_attn = nnx.MultiHeadAttention(self.hidden_size, num_heads=3, batch_first=True, rngs=rngs)
        
        # LSTM
        self.lstmcell = nnx.LSTMCell(self.hidden_size, self.hidden_size, rngs=rngs)
        
        # value
        self.num_bins = 50
        self.value = nnx.Linear(self.hidden_size, self.num_bins, rngs=rngs)
        
        # policy
        self.logit_act = nnx.Linear(self.hidden_size, int(logit_act_shape[-1]), rngs=rngs)
        self.logit_move = nnx.Linear(self.hidden_size, int(logit_move_shape[-1]), rngs=rngs)
        self.logit_target = nnx.Linear(self.hidden_size, int(logit_target_shape[-1]), rngs=rngs)

    @property
    def bins(self):
        """Property to dynamically create bins using self.num_bins."""
        
        return jnp.linspace(-20, 20, self.num_bins)

    @staticmethod
    def load_model_weight(args, flax_model, device=jax.devices("cpu")[0]):
        """
        Load model weights for a Flax model.
        """
        
        model_dir = Path(args.model_dir)
        
        model_dirs = list(model_dir.glob(f"{args.algo}_*"))
        if not model_dirs:
            print(f"No model directories found under {model_dir} with prefix {args.algo}")
            return flax_model, None

        # # Extract idx values and find the highest one
        # def extract_idx(path):
        #     try:
        #         return int(path.name.split("_")[-1])  # Extract idx from the directory name
        #     except ValueError:
        #         return -1  # Invalid directory format, ignore this directory
        
        model_dirs = sorted(model_dirs, key=extract_file_num, reverse=True)
        highest_idx_dir = model_dirs[0]  # Directory with the highest idx
        load_path = highest_idx_dir.resolve()

        abstract_model = nnx.eval_shape(lambda: flax_model)
        graphdef, abstract_state = nnx.split(abstract_model)

        restored_pure_dict, overall_model_states = load_state(abstract_state.to_pure_dict(), load_path)
    
        # checkpointer = ocp.StandardCheckpointer()
        # restored_pure_dict = checkpointer.restore(load_path)
    
        # # Reset RNGs in the restored state
        # # TODO: 버그인가..? LSTMCell에서 쥐고 있는 rngs는 저장할 수 없다는 에러가 발생함.
        # # -> 이걸 해결하기 위해... 모델 로딩 시점에서 억지로 재생성 함
        # restored_pure_dict["model_state_dict"] = reset_rngs_in_state(abstract_state.to_pure_dict(), restored_pure_dict["model_state_dict"])
        
        abstract_state.replace_by_pure_dict(restored_pure_dict)
        
        # Move the model state to the desired device
        abstract_state_ = jax.device_put(abstract_state, device)
        
        model = nnx.merge(graphdef, abstract_state_)
        
        print(f"[load_nnx_model_state] Loaded data from {load_path}")
        return model, overall_model_states

    @staticmethod
    def save_model_weight(path, idx, scale, flax_model, optim_pure_dict):
        """
        Save model weights for a Flax model.
        """
        
        model_dir = Path(path).resolve()
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        
        _, model_state = nnx.split(flax_model)
        # nnx.display(state)
        pure_dict_state = model_state.to_pure_dict()
        # pure_dict_state = remove_keys_from_state(pure_dict_state, {"rngs"}) # TODO: 버그인가..? LSTMCell에서 쥐고 있는 rngs는 저장할 수 없다는 에러가 발생함.
        
        # Prepare data to save
        to_save = {
            "model_pure_dict": pure_dict_state,
            "log_idx": idx,
            "scale": scale,
            "optim_pure_dict": optim_pure_dict,
        }

        # # Save checkpoint
        # checkpointer = ocp.StandardCheckpointer()
        # checkpointer.save(model_dir, to_save)
        # checkpointer.wait_until_finished()
        save_state(to_save, model_dir)
        print(f"[save_nnx_model_state] Saved model state to {model_dir}")
        
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

    def dict_ordering(
        self, item_dict: Dict[str, Union[jnp.ndarray, jnp.ndarray]], is_act: bool = False
    ):
        if is_act: # act-dict
            order_str = ["on_select_act", "on_select_move", "on_select_target", "act_sampled", "move_sampled", "target_sampled"]

        else: # obs-dict
            order_str = ["obs_mine", "obs_ally", "obs_enemy", "avail_act", "avail_move", "avail_target"]

        return tuple(jnp.array(item_dict[key]) for key in order_str if key in item_dict)
    
    def body_encode(self, obs_tuple: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        obs_mine, obs_ally, obs_enemy = obs_tuple
        sym_obs_mine = Symlog(obs_mine)
        sym_obs_ally = Symlog(obs_ally)
        sym_obs_enemy = Symlog(obs_enemy)
        
        # # 죽은 유닛에 대한 마스크 생성, obs_* -> all zero value array
        # dead_units_mask = (sym_obs_mine.sum(dim=-1) == 0) & (sym_obs_ally.sum(dim=-1) == 0) & (sym_obs_enemy.sum(dim=-1) == 0)

        ec_mine = self.norm_mine(nnx.relu(self.encode_mine(sym_obs_mine)))
        ec_ally = self.encode_ally(sym_obs_ally)
        ec_enemy = self.encode_enemy(sym_obs_enemy)

        # mine_v = F.relu(self.mine_attn_v(sym_obs_mine))
        ec_body = self.norm_body(
            nnx.relu(self.encode_body(jnp.concatenate([ec_mine, ec_ally, ec_enemy], axis=-1)))
        )

        # attn_out, attn_weights = self.multihead_attn(ec_mine, ec_body, mine_v, key_padding_mask=dead_units_mask.float()) # q, k, v
        # return attn_out, attn_weights
        return self.norm_attn(nnx.relu(self.encode_attn(ec_body)))

    def act(
        self, obs_dict: Dict[str, jnp.ndarray], hx: jnp.ndarray, cx: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        obs_tuple = self.dict_ordering(obs_dict)
        return self._act(obs_tuple, hx, cx)

    def _act(
        self,
        obs_tuple: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        hx: jnp.ndarray,
        cx: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        out_encode = self.body_encode(obs_tuple[:3])
        (cx, hx), out_hx = self.lstmcell((cx, hx), out_encode)

        logit_act = self.logit_act(out_hx)
        logit_move = self.logit_move(out_hx)
        logit_target = self.logit_target(out_hx)

        act_outs = get_act_outs(
            logit_act, logit_move, logit_target, obs_tuple[3:], jax.random.PRNGKey(0),
        )
        
        out_dict = {
            "act_sampled": jnp.expand_dims(act_outs[0], -1),
            "move_sampled": jnp.expand_dims(act_outs[1], -1),
            "target_sampled": jnp.expand_dims(act_outs[2], -1),
            "logit_act": act_outs[3],
            "logit_move": act_outs[4],
            "logit_target": act_outs[5],
            "hx": hx,
            "cx": cx,
        }
        return out_dict

    def forward(
        self,
        obs_dict: Dict[str, jnp.ndarray],
        act_dict: Dict[str, jnp.ndarray],
        hx: jnp.ndarray,
        cx: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        obs_tuple = self.dict_ordering(obs_dict)
        act_tuple = self.dict_ordering(act_dict, is_act=True)
        return self._forward(obs_tuple, act_tuple, hx, cx)

    def _forward(
        self,
        obs_tuple: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        act_tuple: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        hx: jnp.ndarray,
        cx: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        on_select_act, on_select_move, on_select_target = act_tuple[:3]
        act_sampled, move_sampled, target_sampled = act_tuple[-3:]

        out_encode = self.body_encode(obs_tuple[:3])
        B, S, _ = out_encode.shape

        output = []
        for i in range(S):
            (cx, hx), out_hx = self.lstmcell((cx, hx), out_encode[:, i])
            output.append(out_hx) # hidden state (hx)를 저장
        output = jnp.stack(output, axis=1)

        value = self.value(output)

        logit_act = self.logit_act(output)
        logit_move = self.logit_move(output)
        logit_target = self.logit_target(output)

        log_probs, entropy = get_forward_outs(
            logit_act,
            logit_move,
            logit_target,
            obs_tuple[3:],
            on_select_act,
            on_select_move,
            on_select_target,
            act_sampled,
            move_sampled,
            target_sampled,
        )
        return log_probs.reshape(B, S, -1), entropy.reshape(B, S, -1), value.reshape(B, S, -1)
    
    
@nnx.jit
def jax_model_forward(model: ModelSingle, obs_dict, act_dict, hx, cx):
    return model.forward(obs_dict, act_dict, hx, cx)


@nnx.jit
def jax_model_act(model: ModelSingle, obs_dict, hx, cx):
    return model.act(obs_dict, hx, cx)