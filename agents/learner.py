import zmq
import optax
import jax
import jax.numpy as jnp

from flax import nnx

import asyncio
import math

import zmq.asyncio
from collections import defaultdict

from utils.lock import Mutex, LockManager
from utils.utils import (
    Protocol,
    encode,
    decode,
    ExecutionTimer,
    Params,
    # extract_values,
    select_least_used_jax_gpu,
)

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
        model_cls : "ModelSingle",
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
        
        # Select device
        device = select_least_used_jax_gpu()
        # device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0]
        self.args.device = device if device else jax.devices("cpu")[0] # 기본적으로 Learner 쪽은 Cuda 디바이스
        jax.config.update('jax_default_device', self.args.device)

        print(f"learner device: {self.args.device}")

        self.device = self.args.device
        self.idx = 0
        self.scale = jnp.nan # 초기화
        
        #TODO: manual 코드
        self.metrics = nnx.MultiMetric(
            total_loss=nnx.metrics.Average('total_loss'),
            policy_loss=nnx.metrics.Average('policy_loss'),
            value_loss=nnx.metrics.Average('value_loss'),
            policy_entropy=nnx.metrics.Average('policy_entropy'),
            scale=nnx.metrics.Average('scale'),
            min_ratio=nnx.metrics.Average('min_ratio'),
            max_ratio=nnx.metrics.Average('max_ratio'),
            avg_ratio=nnx.metrics.Average('avg_ratio'),
        )

        model = model_cls(self.args, self.env_space)

        self.model, overall_model_states = model_cls.load_model_weight(self.args, model, self.device)
        if overall_model_states is not None:
            self.idx = overall_model_states["log_idx"]
            self.scale = overall_model_states["scale"]

        tx = optax.chain(
            optax.clip_by_global_norm(self.args.max_grad_norm),
            optax.adam(learning_rate=self.args.lr),
        )
        self.optimizer = nnx.Optimizer(model, tx)
        if overall_model_states is not None:
            optim_graphdef, optim_state = nnx.split(self.optimizer)
            optim_state.replace_by_pure_dict(overall_model_states["optim_pure_dict"])
            nnx.update(self.optimizer, optim_state)

        self.zeromq_set(learner_ip, learner_worker_port)

        from flax.metrics import tensorboard
        self.writer = tensorboard.SummaryWriter(log_dir=self.args.result_dir)  # tensorboard-log
        
    def __del__(self):  # 소멸자
        if hasattr(self, "pub_socket"):
            self.pub_socket.close()
        if hasattr(self, "sub_socket"):
            self.sub_socket.close()
        if hasattr(self, "writer"):
            # self.writer.flush()
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

    def pub_model(self, model_pure_dict):  # learner -> worker
        self.pub_socket.send_multipart([*encode(Protocol.Model, model_pure_dict)])

    async def log_loss_tensorboard(self, timer: ExecutionTimer):
        for k, v in self.metrics.compute().items():
            self.writer.scalar(k, v, self.idx)

        if timer is not None and isinstance(timer, ExecutionTimer):
            for k, v in timer.timer_dict.items():
                self.writer.scalar(
                    f"{k}-elapsed-mean-sec", sum(v) / (len(v) + 1e-6), self.idx
                )
            for k, v in timer.throughput_dict.items():
                self.writer.scalar(
                    f"{k}-transition-per-secs", sum(v) / (len(v) + 1e-6), self.idx
                )

        if self.stat_q.qsize() > 0:
            stat_dict = await self.stat_q.get()
            for k, v in stat_dict.items():
                if k == "epi_rev_vec":
                    tag = f"stat-{k}"
                    y = jnp.mean(v)
                    self.writer.scalar(tag, y, self.idx)

            _mean_rev_vec = jnp.mean(stat_dict["epi_rev_vec"], axis=0)

            for rdx, (r_param, weight) in enumerate(REWARD_PARAM.items()):
                tag = f"mean-weighted-reward-{r_param}"
                weighted_reward = _mean_rev_vec[rdx] * weight
                self.writer.scalar(tag, weighted_reward, self.idx)

        self.metrics.reset()

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
                batch_dict[space_name][k] = getattr(self, f"sh_{k}").reshape((B, S, D))

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
                
                jnp_tensor_data = getattr(shm_inf, attr_name)[:math.prod((N, *v.nvec[1:]))].reshape(-1, S, D)

                if k in batch_dict[space_name]:
                    # 기존 jnp 배열의 Batch 축에 concatenate
                    batch_dict[space_name][k] = jnp.concatenate([jnp_tensor_data, batch_dict[space_name][k]], axis=0)
                else:
                    batch_dict[space_name][k] = jnp_tensor_data

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