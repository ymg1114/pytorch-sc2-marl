import os
import zmq
import torch
# import time
# import torch.nn.functional as F
import asyncio
import math

# import torch.nn.functional as F
# import multiprocessing as mp
import torch.jit as jit
import numpy as np
from collections import defaultdict, deque
from torch.distributions import Categorical, Uniform
# from functools import partial

import zmq.asyncio

from utils.lock import Mutex, LockManager
from utils.utils import (
    Protocol,
    encode,
    decode,
    # make_gpu_batch,
    ExecutionTimer,
    Params,
    to_torch,
    # extract_values,
)
from torch.optim import Adam, RMSprop

from abc import ABC, abstractmethod
from .storage_module.shared_batch import SMInterface
from . import (
    ppo_awrapper,
    impala_awrapper,
)
from rewarder.rewarder import REWARD_PARAM

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from networks.network import ModelSingle


timer = ExecutionTimer(
    num_transition=Params.seq_len * Params.batch_size * 1
)  # Learner에서 데이터 처리량 (학습)


class LearnerBase(ABC):
    def __init__(
        self,
        args,
        mutex,
        model_cls,
        shm_ref,
        lock_manager,
        stop_event,
        learner_ip,
        learner_worker_port,
        env_space,
        heartbeat=None,
    ):
        self.args = args
        self.env_space = env_space
        self.mutex = mutex
        self.stop_event = stop_event
        
        self.shm_ref = shm_ref
        self.lock_manager: LockManager = lock_manager
        
        self.heartbeat = heartbeat
        
        self.device = self.args.device
        self.idx = 0
        self.scale = None
        
        model: "ModelSingle" = model_cls(self.args, self.env_space)
        self.model = jit.script(model).to(self.device)
        
        out_dict = model_cls.set_model_weight(self.args, self.device)
        if out_dict is not None:
            self.model.load_state_dict(out_dict["state_dict"])
            self.idx = out_dict["log_idx"]
            self.scale = out_dict["scale"]
            
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        # self.optimizer = RMSprop(self.model.parameters(), lr=self.args.lr, eps=1e-5)
        if out_dict is not None:
            self.optimizer.load_state_dict(out_dict["optim_state_dict"])
            
        self.CT = Categorical

        # self.to_gpu = partial(make_gpu_batch, device=self.device)
        self.zeromq_set(learner_ip, learner_worker_port)
        
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=args.result_dir)  # tensorboard-log
        
    def __del__(self):  # 소멸자
        if hasattr(self, "pub_socket"):
            self.pub_socket.close()
        if hasattr(self, "sub_socket"):
            self.sub_socket.close()
        if hasattr(self, "writer"):
            self.writer.close()
            
    def zeromq_set(self, learner_ip, learner_worker_port):
        acontext = zmq.asyncio.Context()
        
        # worker <-> learner
        self.sub_socket = acontext.socket(zmq.SUB) # subscribe stat-data
        self.sub_socket.bind(
            f"tcp://{learner_ip}:{int(learner_worker_port) + 2}"
        )
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"") 
        
        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(
            f"tcp://{learner_ip}:{int(learner_worker_port) + 1}"
        )  # publish fresh learner-model

    def pub_model(self, model_state_dict):  # learner -> worker
        self.pub_socket.send_multipart([*encode(Protocol.Model, model_state_dict)])

    async def log_loss_tensorboard(self, timer: ExecutionTimer, loss, detached_losses):
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

        if "scale" in detached_losses:
            self.writer.add_scalar(
                "actor-loss-eval-scale", detached_losses["scale"], self.idx
            )

        if "loss-temperature" in detached_losses:
            self.writer.add_scalar(
                "loss-temperature", detached_losses["loss-temperature"].mean(), self.idx
            )

        if timer is not None and isinstance(timer, ExecutionTimer):
            for k, v in timer.timer_dict.items():
                self.writer.add_scalar(
                    f"{k}-elapsed-mean-sec", sum(v) / (len(v) + 1e-6), self.idx
                )
            for k, v in timer.throughput_dict.items():
                self.writer.add_scalar(
                    f"{k}-transition-per-secs", sum(v) / (len(v) + 1e-6), self.idx
                )

        if self.stat_q.qsize() > 0:
            stat_dict = await self.stat_q.get()
            # stat_keys = list(sample_stat_dict.keys())

            for k, v in stat_dict.items():
                if k != "epi_rew_vec":
                    tag = f"stat-{k}"
                    y = np.mean(v)
                    self.writer.add_scalar(tag, y, self.idx)

            _mean_rew_vec = np.mean(stat_dict["epi_rew_vec"], axis=0)
            
            for rdx, (r_param, weight) in enumerate(REWARD_PARAM.items()):
                tag = f"mean-weighted-reward-{r_param}"
                weighted_reward = _mean_rew_vec[rdx] * weight
                self.writer.add_scalar(tag, weighted_reward, self.idx)
                
    # @staticmethod
    # def copy_to_ndarray(src):
    #     dst = np.empty(src.shape, dtype=src.dtype)
    #     np.copyto(
    #         dst, src
    #     )  # 학습용 데이터를 새로 생성하고, 공유메모리의 데이터 오염을 막기 위함.
    #     return dst
        
    @ppo_awrapper(timer=timer)
    def learning_ppo(self): ...

    @impala_awrapper(timer=timer)
    def learning_impala(self): ...

    async def sub_stat_data(self):
        while not self.stop_event.is_set():
            protocol, data = decode(*await self.sub_socket.recv_multipart())
            assert protocol is Protocol.Stat

            if self.stat_q.full():
                print("stat_q is full, consuming an item before putting new one")
                await self.stat_q.get()
                
            await self.stat_q.put(data)
            print("stat-data is received !")
            await asyncio.sleep(1e-4)
    
    async def learning_chain_ppo(self):
        self.batch_queue = asyncio.Queue(1024)
        self.stat_q = asyncio.Queue(128)
        tasks = [
            asyncio.create_task(self.learning_ppo()),
            asyncio.create_task(self.sub_stat_data()),
            asyncio.create_task(self.put_batch_to_batch_q()),
        ]
        await asyncio.gather(*tasks)

    async def learning_chain_impala(self):
        self.batch_queue = asyncio.Queue(1024)
        self.stat_q = asyncio.Queue(128)
        tasks = [
            asyncio.create_task(self.learning_impala()),
            asyncio.create_task(self.sub_stat_data()),
            asyncio.create_task(self.put_batch_to_batch_q()),
        ]
        await asyncio.gather(*tasks)

    @abstractmethod
    def is_sh_ready(self, *args, **kwargs):
        ...

    @abstractmethod
    def sample_batch_from_sh_memory(self, *args, **kwargs):
        ...
            
    @abstractmethod
    async def put_batch_to_batch_q(self, *args, **kwargs):
        ...
        

class LearnerSingle(LearnerBase, SMInterface):
    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        SMInterface.__init__(self, shm_ref=self.shm_ref, env_space=self.env_space)
                
    def is_sh_ready(self):
        Bat = self.args.batch_size
        Shn = self.sh_data_num
        return True if Shn.value >= Bat else False
        
    def sample_batch_from_sh_memory(self):
        batch_dict = defaultdict(dict)

        def _extract_batch(space_name, space):
            for k, v in space.items():
                assert hasattr(self, f"sh_{k}")
                B, S, D = v.nvec  # Batch, Sequence, Dim
                batch_dict[space_name][k] = to_torch(getattr(self, f"sh_{k}").reshape((B, S, D)))

        _extract_batch("obs", self.env_space["obs"])
        _extract_batch("act", self.env_space["act"])
        _extract_batch("rew", self.env_space["rew"])
        _extract_batch("info", self.env_space["info"])

        return batch_dict
        
    async def put_batch_to_batch_q(self):
        while not self.stop_event.is_set():
            assert isinstance(self.mutex, Mutex)
            
            with self.mutex.lock():
                if self.is_sh_ready():
                    batch_dict = self.sample_batch_from_sh_memory()
                    await self.batch_queue.put(batch_dict)
                    self.reset_data_num()  # 공유메모리 저장 인덱스 (batch_num) 초기화
                    print("batch is ready !")

            await asyncio.sleep(1e-4)
            
        
class LearnerMulti(LearnerBase):
    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lock_manager.reconnect_shared_memory(self.env_space)
        
    def is_sh_ready(self):
        Bat = self.args.batch_size
        total_count = sum(i[1].get_count() for i in self.lock_manager.shm_mutexes)
        return True if total_count >= Bat else False

    def sample_batch_from_sh_memory(self):
        batch_dict = defaultdict(dict)
            
        def _extract_batch(space_name, shm_inf, space):
            for k, v in space.items():
                # 데이터 조회 및 변환
                attr_name = f"sh_{k}"
                if not hasattr(shm_inf, attr_name):
                    raise AttributeError(f"{shm_inf} does not have attribute {attr_name}")
                
                _, S, D = v.nvec  # Batch, Sequence, Dim
                N = shm_inf.get_count()
                
                tensor_data = to_torch(getattr(shm_inf, attr_name)[:math.prod((N, *v.nvec[1:]))]).reshape(-1, S, D)

                if k in batch_dict[space_name]:
                    # 기존 torch 배열의 Batch 축에 concatenate
                    batch_dict[space_name][k] = torch.cat([tensor_data, batch_dict[space_name][k]], dim=0)
                else:
                    batch_dict[space_name][k] = tensor_data

        # 모든 공간에 대해 데이터를 추출
        for shm_lock, shm_inf in self.lock_manager.shm_mutexes:
            for space_name in ["obs", "act", "rew", "info"]:
                _extract_batch(space_name, shm_inf, self.env_space[space_name])

        return batch_dict
        
    async def put_batch_to_batch_q(self):
        while not self.stop_event.is_set():

            with self.lock_manager.Lock():
                if self.is_sh_ready():
                    batch_dict = self.sample_batch_from_sh_memory()
                    await self.batch_queue.put(batch_dict)
                    self.lock_manager.Reset() # 모든 공유메모리 저장 인덱스 (batch_num) 초기화
                    print("batch is ready !")

            await asyncio.sleep(1e-4)
        
        
class LearnerMultiPPO(LearnerMulti): ...


class LearnerMultiIMPALA(LearnerMulti): ...

        
class LearnerSinglePPO(LearnerSingle): ...


class LearnerSingleIMPALA(LearnerSingle): ...