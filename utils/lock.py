import time
import torch.multiprocessing as mp

from contextlib import contextmanager

class Lock:
    def __init__(self):
        self._lock = mp.Lock()

    # 여러 프로세스가 공유 자원에 동시에 접근하는 것을 방지하여 데이터 무결성을 보장
    @contextmanager
    def lock(self):
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()


class Mutex:
    def __init__(self):
        self._mutex = mp.Semaphore(1)

    # 1개 (티겟 1개) 프로세스만 공유 자원에 접근하는 것을 허용.
    @contextmanager
    def lock(self):
        self._mutex.acquire()
        try:
            yield
        finally:
            self._mutex.release()


class LockManager:
    def __init__(self, env_space, num_locks=3):
        from agents.storage_module.shared_batch import (
            setup_shared_memory, SMInterface
        )
        
        self.OverAllLock = mp.Lock()
        
        self.shm_mutexes = []
        for nl in range(num_locks):
            shm = setup_shared_memory(env_space, nl)
            self.shm_mutexes.append([mp.Lock(), shm])
            
        self.current_index = mp.Value('i', 0)  # Round Robin index (공유 변수)

    def reconnect_shared_memory(self, env_space):
        """각 프로세스에서 LockManager의 공유 메모리 다시 연결"""
        
        from agents.storage_module.shared_batch import (
            setup_shared_memory, SMInterface
        )
        
        _tmp = []
        for lock, shm in self.shm_mutexes:
            shm_inf = SMInterface(shm, env_space)
            _tmp.append([lock, shm_inf])
        self.shm_mutexes = _tmp # 오버라이드 (덮어쓰기)
        
    def Reset(self):
        [i[1].reset_data_num() for i in self.shm_mutexes]
        
    @contextmanager
    def Lock(self):
        self.OverAllLock.acquire()
        try:
            yield
        finally:
            self.OverAllLock.release()

    @contextmanager
    def acquire_next_available_lock_and_shm(self, max_retries=3, retry_interval=0.5):
        """
        Round Robin 방식으로 락을 획득하며, 실패 시 재시도.
        :param max_retries: 최대 재시도 횟수
        :param retry_interval: 재시도 간격 (초)
        """
        total_locks = len(self.shm_mutexes)
        retries = 0

        while retries <= max_retries:
            acquired_mutex_shm = None

            for _ in range(total_locks):  # 모든 Lock을 순회
                with self.current_index.get_lock():  # Round Robin 인덱스 동기화
                    index = self.current_index.value
                    self.current_index.value = (self.current_index.value + 1) % total_locks

                mutex_shm = self.shm_mutexes[index]
                if mutex_shm[0].acquire(block=False):  # 락 획득 시도
                    acquired_mutex_shm = mutex_shm
                    break

            if acquired_mutex_shm is not None:
                try:
                    yield acquired_mutex_shm[1]  # SharedMemory-Interface 객체 반환
                finally:
                    acquired_mutex_shm[0].release()  # 락 해제
                return  # 락을 성공적으로 반환했으므로 종료

            # 락 획득 실패, 재시도
            retries += 1
            time.sleep(retry_interval)

        # 최대 재시도 초과 시 예외 발생
        raise RuntimeError("All locks are busy. Could not acquire any lock after retries.")
