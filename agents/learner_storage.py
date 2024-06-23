import time
import zmq
import zmq.asyncio
import asyncio

import numpy as np
# import multiprocessing as mp

from .storage_module.shared_batch import SMInterface

from buffers.rollout_assembler import RolloutAssembler
from utils.lock import Mutex
from utils.utils import (
    Protocol,
    mul,
    decode,
    flatten,
    counted,
)
from rewarder.rewarder import REWARD_PARAM

# timer = ExecutionTimer(num_transition=Params.seq_len*1) # LearnerStorage에서 데이터 처리량 (수신) / 부정확한 값이지만 어쩔 수 없음


class LearnerStorage(SMInterface):
    def __init__(
        self,
        args,
        mutex,
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
        self.stop_event = stop_event

        self.mutex: Mutex = mutex

        if shared_stat_array is not None:
            self.np_shared_stat_array: np.ndarray = np.frombuffer(
                buffer=shared_stat_array.get_obj(), dtype=np.float32, count=-1
            )

        self.heartbeat = heartbeat

        self.zeromq_set(learner_ip, learner_port)
        self.get_shared_memory_interface()

    def __del__(self):  # 소멸자
        if hasattr(self, "sub_socket"):
            self.sub_socket.close()

    def zeromq_set(self, learner_ip, learner_port):
        context = zmq.asyncio.Context()

        # learner-storage <-> manager
        self.sub_socket = context.socket(zmq.SUB)  # subscribe batch-data, stat-data
        self.sub_socket.bind(f"tcp://{learner_ip}:{learner_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

    async def shared_memory_chain(self):
        self.rollout_assembler = RolloutAssembler(self.args, asyncio.Queue(1024))

        tasks = [
            asyncio.create_task(self.retrieve_rollout_from_worker()),
            asyncio.create_task(self.build_as_batch()),
        ]
        await asyncio.gather(*tasks)

    async def retrieve_rollout_from_worker(self):
        while not self.stop_event.is_set():
            protocol, data = decode(*await self.sub_socket.recv_multipart())
            if protocol is Protocol.Rollout:
                await self.rollout_assembler.push(data)

            elif protocol is Protocol.Stat:
                self.log_stat_tensorboard(**data)
            else:
                assert False, f"Wrong protocol: {protocol}"

            await asyncio.sleep(0.001)

    async def build_as_batch(self):
        while not self.stop_event.is_set():
            if self.heartbeat is not None:
                self.heartbeat.value = time.time()

            # with timer.timer("learner-storage-throughput", check_throughput=True):
            data = await self.rollout_assembler.pop()
            self.make_batch(data)
            print("rollout is poped !")

            await asyncio.sleep(0.001)

    @counted
    def log_stat_tensorboard(self, **data):
        _len = data["log_len"]
        _mean_battle_won = data["mean_battle_won"]
        _mean_dead_allies = data["mean_dead_allies"]
        _mean_dead_enemies = data["mean_dead_enemies"]
        _mean_rew_vec = data["mean_rew_vec"]

        x = self.log_stat_tensorboard.calls * _len  # global game counts

        # TODO: 좋은 구조는 아님...
        if self.np_shared_stat_array is not None:
            assert self.np_shared_stat_array.size == len(REWARD_PARAM) + 2 + 3

            self.np_shared_stat_array[0] = x  # global game counts
            self.np_shared_stat_array[1] = 1  # 기록 가능 활성화 (activate)

            self.np_shared_stat_array[2] = _mean_battle_won  # mean_battle_won
            self.np_shared_stat_array[3] = _mean_dead_allies  # mean_dead_allies
            self.np_shared_stat_array[4] = _mean_dead_enemies  # mean_dead_enemies
            
            for rdx, (r_parma, weight) in enumerate(REWARD_PARAM.items()):
                weighted_reward = _mean_rew_vec[rdx] * weight
                self.np_shared_stat_array[rdx+5] = weighted_reward  # rew_vec~

    def make_batch(self, rollout):
        Bat = self.args.batch_size
        N = self.sh_data_num.value

        if N < Bat:
            def _acquire(key, value):
                assert hasattr(self, f"sh_{key}")
                assert key in rollout
                
                B, S, D = value.nvec # Batch, Sequence, Dim
                assert B == Bat
                return flatten(rollout[key])

            def _update_shared_memory(space):
                for k, v in space.items():
                    B, S, D = v.nvec
                    getattr(self, f"sh_{k}")[S*N*D: S*(N+1)*D] = _acquire(k, v)

            _update_shared_memory(self.env_space["obs"])
            _update_shared_memory(self.env_space["act"])
            _update_shared_memory(self.env_space["rew"])
            _update_shared_memory(self.env_space["info"])

            self.sh_data_num.value += 1