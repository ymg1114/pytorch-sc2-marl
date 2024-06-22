import os
import zmq
import time
import torch
import torch.nn.functional as F
import asyncio

# import torch.nn.functional as F
# import multiprocessing as mp
import numpy as np
from collections import defaultdict
from torch.distributions import Categorical, Uniform
from functools import partial

from utils.lock import Mutex
from utils.utils import (
    Protocol,
    encode,
    make_gpu_batch,
    ExecutionTimer,
    Params,
    to_torch,
)
from torch.optim import Adam, RMSprop

from .storage_module.shared_batch import SMInterFace
from . import (
    ppo_awrapper,
    impala_awrapper,
)
from rewarder.rewarder import REWARD_PARAM

timer = ExecutionTimer(
    num_transition=Params.seq_len * Params.batch_size * 1
)  # Learner에서 데이터 처리량 (학습)


class LearnerBase(SMInterFace):
    def __init__(
        self,
        args,
        mutex,
        model,
        shm_ref,
        stop_event,
        learner_ip,
        learner_port,
        env_space,
        shared_stat_array=None,
        heartbeat=None,
    ):
        super().__init__(shm_ref=shm_ref, env_space=env_space)
        self.args = args
        self.env_space = env_space
        self.mutex: Mutex = mutex
        self.stop_event = stop_event

        if shared_stat_array is not None:
            self.np_shared_stat_array: np.ndarray = np.frombuffer(
                buffer=shared_stat_array.get_obj(), dtype=np.float32, count=-1
            )

        self.heartbeat = heartbeat

        self.device = self.args.device
        self.model = model.to(self.device)

        # self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.args.lr, eps=1e-5)
        self.CT = Categorical

        self.to_gpu = partial(make_gpu_batch, device=self.device)

        self.zeromq_set(learner_ip, learner_port)
        self.get_shared_memory_interface()
        from tensorboardX import SummaryWriter

        self.writer = SummaryWriter(log_dir=args.result_dir)  # tensorboard-log

    def __del__(self):  # 소멸자
        if hasattr(self, "pub_socket"):
            self.pub_socket.close()

    def zeromq_set(self, learner_ip, learner_port):
        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(
            f"tcp://{learner_ip}:{int(learner_port) + 1}"
        )  # publish fresh learner-model

    def pub_model(self, model_state_dict):  # learner -> worker
        self.pub_socket.send_multipart([*encode(Protocol.Model, model_state_dict)])

    def log_loss_tensorboard(self, timer: ExecutionTimer, loss, detached_losses):
        self.writer.add_scalar("total-loss", float(loss.item()), self.idx)
        if "value-loss" in detached_losses:
            self.writer.add_scalar(
                "original-value-loss", detached_losses["value-loss"], self.idx
            )

        if "policy-loss" in detached_losses:
            self.writer.add_scalar(
                "original-policy-loss", detached_losses["policy-loss"], self.idx
            )

        if "policy-entropy" in detached_losses:
            self.writer.add_scalar(
                "original-policy-entropy", detached_losses["policy-entropy"], self.idx
            )

        if "ratio" in detached_losses:
            self.writer.add_scalar(
                "min-ratio", detached_losses["ratio"].min(), self.idx
            )
            self.writer.add_scalar(
                "max-ratio", detached_losses["ratio"].max(), self.idx
            )
            self.writer.add_scalar(
                "avg-ratio", detached_losses["ratio"].mean(), self.idx
            )

        if "loss-temperature" in detached_losses:
            self.writer.add_scalar(
                "loss-temperature", detached_losses["loss-temperature"].mean(), self.idx
            )

        if "loss_alpha" in detached_losses:
            self.writer.add_scalar(
                "loss_alpha", detached_losses["loss_alpha"].mean(), self.idx
            )

        if "alpha" in detached_losses:
            self.writer.add_scalar("alpha", detached_losses["alpha"], self.idx)

        # TODO: 좋은 구조는 아님...
        if self.np_shared_stat_array is not None:
            assert self.np_shared_stat_array.size == len(REWARD_PARAM) + 2
            if (
                bool(self.np_shared_stat_array[1]) is True
            ):  # 기록 가능 활성화 (activate)

                x = self.np_shared_stat_array[0]  # global game counts
                for rdx, (r_parma, weight) in enumerate(REWARD_PARAM.items()):
                    y = self.np_shared_stat_array[rdx+2]
                    tag = f"50-game-mean-rew-stat-of-{r_parma}"
                    self.writer.add_scalar(tag, y, x)
                    
                self.np_shared_stat_array[1] = 0  # 기록 가능 비활성화 (deactivate)
                
        if timer is not None and isinstance(timer, ExecutionTimer):
            for k, v in timer.timer_dict.items():
                self.writer.add_scalar(
                    f"{k}-elapsed-mean-sec", sum(v) / (len(v) + 1e-6), self.idx
                )
            for k, v in timer.throughput_dict.items():
                self.writer.add_scalar(
                    f"{k}-transition-per-secs", sum(v) / (len(v) + 1e-6), self.idx
                )

    # @staticmethod
    # def copy_to_ndarray(src):
    #     dst = np.empty(src.shape, dtype=src.dtype)
    #     np.copyto(
    #         dst, src
    #     )  # 학습용 데이터를 새로 생성하고, 공유메모리의 데이터 오염을 막기 위함.
    #     return dst

    def sample_batch_from_sh_memory(self):
        # sq = self.args.seq_len
        # # bn = self.args.batch_size
        # hs = self.args.hidden_size
        # ac = self.args.action_space

        # _sh_obs_batch = self.sh_obs_batch.reshape((num_buf, sq, *sha))[idx]
        # _sh_act_batch = self.sh_act_batch.reshape((num_buf, sq, 1))[idx]
        # _sh_rew_batch = self.sh_rew_batch.reshape((num_buf, sq, 1))[idx]
        # _sh_logits_batch = self.sh_logits_batch.reshape((num_buf, sq, ac))[idx]
        # _sh_log_prob_batch = self.sh_log_prob_batch.reshape((num_buf, sq, 1))[idx]
        # _sh_is_fir_batch = self.sh_is_fir_batch.reshape((num_buf, sq, 1))[idx]
        # _sh_hx_batch = self.sh_hx_batch.reshape((num_buf, sq, hs))[idx]
        # _sh_cx_batch = self.sh_cx_batch.reshape((num_buf, sq, hs))[idx]

        # # (batch, seq, feat)
        # sh_obs_bat = LearnerBase.copy_to_ndarray(_sh_obs_batch)
        # sh_act_bat = LearnerBase.copy_to_ndarray(_sh_act_batch)
        # sh_rew_bat = LearnerBase.copy_to_ndarray(_sh_rew_batch)
        # sh_logits_bat = LearnerBase.copy_to_ndarray(_sh_logits_batch)
        # sh_log_prob_bat = LearnerBase.copy_to_ndarray(_sh_log_prob_batch)
        # sh_is_fir_bat = LearnerBase.copy_to_ndarray(_sh_is_fir_batch)
        # sh_hx_bat = LearnerBase.copy_to_ndarray(_sh_hx_batch)
        # sh_cx_bat = LearnerBase.copy_to_ndarray(_sh_cx_batch)
        
        batch_dict = defaultdict(dict)

        obs_space = self.env_space["obs"]
        act_space = self.env_space["act"]
        rew_space = self.env_space["rew"]
        info_space = self.env_space["info"]
        
        for k, v in obs_space.items():
            assert hasattr(self, f"sh_{k}")
            B, S, D = v.nvec # Batch, Sequence, Dim
            batch_dict["obs"][k] = to_torch(getattr(self, f"sh_{k}").reshape((B, S, D)))
       
        for k, v in act_space.items():
            assert hasattr(self, f"sh_{k}")
            B, S, D = v.nvec # Batch, Sequence, Dim
            batch_dict["act"][k] = to_torch(getattr(self, f"sh_{k}").reshape((B, S, D)))

        for k, v in rew_space.items():
            assert hasattr(self, f"sh_{k}")
            B, S, D = v.nvec # Batch, Sequence, Dim
            batch_dict["rew"][k] = to_torch(getattr(self, f"sh_{k}").reshape((B, S, D)))
            
        for k, v in info_space.items():
            assert hasattr(self, f"sh_{k}")
            B, S, D = v.nvec # Batch, Sequence, Dim
            batch_dict["info"][k] = to_torch(getattr(self, f"sh_{k}").reshape((B, S, D)))
       
        return batch_dict
    
    @ppo_awrapper(timer=timer)
    def learning_ppo(self): ...

    @impala_awrapper(timer=timer)
    def learning_impala(self): ...

    def is_sh_ready(self):
        bn = self.args.batch_size
        val = self.sh_data_num.value
        return True if val >= bn else False

    async def put_batch_to_batch_q(self):
        while not self.stop_event.is_set():
            if self.is_sh_ready():
                batch_dict = self.sample_batch_from_sh_memory()
                await self.batch_queue.put(batch_dict)
                self.reset_data_num()  # 공유메모리 저장 인덱스 (batch_num) 초기화
                print("batch is ready !")

            await asyncio.sleep(0.001)

    async def learning_chain_ppo(self):
        self.batch_queue = asyncio.Queue(1024)
        tasks = [
            asyncio.create_task(self.learning_ppo()),
            asyncio.create_task(self.put_batch_to_batch_q()),
        ]
        await asyncio.gather(*tasks)

    async def learning_chain_impala(self):
        self.batch_queue = asyncio.Queue(1024)
        tasks = [
            asyncio.create_task(self.learning_impala()),
            asyncio.create_task(self.put_batch_to_batch_q()),
        ]
        await asyncio.gather(*tasks)


class LearnerSinglePPO(LearnerBase): ...


class LearnerSingleIMPALA(LearnerSinglePPO): ...