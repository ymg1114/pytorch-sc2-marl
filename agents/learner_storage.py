import time
import zmq
import zmq.asyncio
import asyncio

import numpy as np
# import multiprocessing as mp

from abc import ABC, abstractmethod
from .storage_module.shared_batch import SMInterface

from buffers.rollout_assembler import RolloutAssembler
from utils.lock import Mutex, LockManager
from utils.utils import (
    Protocol,
    # mul,
    decode,
    flatten,
    # counted,
)


class LearnerStorageBase(ABC):
    def __init__(
        self,
        args,
        mutex,
        shm_ref,
        lock_manager,
        stop_event,
        learner_ip,
        learner_port,
        env_space,
        heartbeat=None,
    ):
        self.args = args
        self.env_space = env_space
        self.stop_event = stop_event
        
        self.shm_ref = shm_ref
        self.lock_manager: LockManager = lock_manager
        
        self.mutex: Mutex = mutex

        self.heartbeat = heartbeat

        self.zeromq_set(learner_ip, learner_port)

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
            asyncio.create_task(self.retrieve_rollout_from_manager()),
            asyncio.create_task(self.build_as_batch()),
        ]
        await asyncio.gather(*tasks)

    async def retrieve_rollout_from_manager(self):
        while not self.stop_event.is_set():
            protocol, data = decode(*await self.sub_socket.recv_multipart())
            assert protocol is Protocol.Rollout
            if self.heartbeat is not None:
                self.heartbeat.value = time.monotonic()
            
            await self.rollout_assembler.push(data)
            await asyncio.sleep(1e-4)

    async def build_as_batch(self):
        while not self.stop_event.is_set():
            # with timer.timer("learner-storage-throughput", check_throughput=True):
            trajectory = await self.rollout_assembler.pop()
            with self.mutex.lock():
                self.make_batch(trajectory)
            print("trajectory is poped !")

            await asyncio.sleep(1e-4)

    @abstractmethod
    def make_batch(self, *args, **kwargs):
        ...
        

class LearnerStorageSingle(LearnerStorageBase, SMInterface):
    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        SMInterface.__init__(self, shm_ref=self.shm_ref, env_space=self.env_space)
        
    def make_batch(self, trajectory):
        Bat = self.args.batch_size
        Shn = self.sh_data_num
        
        if Shn.value < Bat:
            N = Shn.value
            
            def _acquire(key, value):
                assert hasattr(self, f"sh_{key}")
                assert key in trajectory
                
                B, S, D = value.nvec # Batch, Sequence, Dim
                assert B == Bat
                
                _T = trajectory[key]
                assert _T.shape == (S, D)
                return flatten(_T)

            def _update_shared_memory(space):
                """공유 메모리에 쓰기 작업 수행. Lock을 도입해
                데이터 무결성 확보
                """
                
                for k, v in space.items():
                    B, S, D = v.nvec # Batch, Sequence, Dim
                    getattr(self, f"sh_{k}")[S*N*D: S*(N+1)*D] = _acquire(k, v)

            _update_shared_memory(self.env_space["obs"])
            _update_shared_memory(self.env_space["act"])
            _update_shared_memory(self.env_space["rew"])
            _update_shared_memory(self.env_space["info"])
            
            with Shn.get_lock(): 
                Shn.value += 1
                
                
class LearnerStorageMulti(LearnerStorageBase):
    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lock_manager.reconnect_shared_memory(self.env_space)
        
    def make_batch(self, trajectory):
        Bat = self.args.batch_size

        # success = False  # 공유메모리 인터페이스 접근 성공 여부 플래그
        # while not success:
        with self.lock_manager.Lock():
            # try:
            with self.lock_manager.acquire_next_available_lock_and_shm(retry_interval=0.05) as shm_inf:
                if shm_inf.sh_data_num.value < Bat:
                    N = shm_inf.sh_data_num.value
                    
                    def _acquire(key, value):
                        assert hasattr(shm_inf, f"sh_{key}")
                        assert key in trajectory
                        
                        B, S, D = value.nvec # Batch, Sequence, Dim
                        assert B == Bat
                        
                        _T = trajectory[key]
                        assert _T.shape == (S, D)
                        return flatten(_T)

                    def _update_shared_memory(space):
                        """공유 메모리에 쓰기 작업 수행. Lock을 도입해
                        데이터 무결성 확보
                        """
                        
                        for k, v in space.items():
                            B, S, D = v.nvec # Batch, Sequence, Dim
                            getattr(shm_inf, f"sh_{k}")[S*N*D: S*(N+1)*D] = _acquire(k, v)
                            
                    _update_shared_memory(self.env_space["obs"])
                    _update_shared_memory(self.env_space["act"])
                    _update_shared_memory(self.env_space["rew"])
                    _update_shared_memory(self.env_space["info"])
                    
                    with shm_inf.sh_data_num.get_lock(): 
                        shm_inf.sh_data_num.value += 1
                    
                        # success = True  # 접근 성공 시 루프 종료
                        
            #     except RuntimeError as e:
            #         # print(f"Task {task_id}: {e}")
            #         time.sleep(0.1)  # 재시도 대기
                    
            # time.sleep(1e-4)
