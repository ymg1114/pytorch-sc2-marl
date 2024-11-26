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
        """LockManager 초기화
        
        각 프로세스의 락 및 공유 메모리를 관리하며, Round Robin 방식으로 락을 획득.
        """
        
        from agents.storage_module.shared_batch import setup_shared_memory
        
        self.OverAllLock = mp.Lock()
        
        self.shm_mutexes = [
            [mp.Lock(), setup_shared_memory(env_space, lock_id)]
            for lock_id in range(num_locks)
        ]
        self.current_index = mp.Value('i', 0)  # Round Robin 인덱스 (공유 변수)

    def reconnect_shared_memory(self, env_space):
        """프로세스에서 공유 메모리 접근 인터페이스 재연결
        
        공유 메모리는 OS 수준에서 전역으로 관리되지만,
        접근 인터페이스는 프로세스 레벨의 객체로, 끊김 발생 시 재생성이 필요.
        """
        
        from agents.storage_module.shared_batch import SMInterface

        # 공유 메모리 인터페이스 재생성 및 덮어쓰기
        self.shm_mutexes = [
            [lock, SMInterface(shm, env_space)]
            for lock, shm in self.shm_mutexes
        ]

    def Reset(self):
        for _, shm_interface in self.shm_mutexes:
            shm_interface.reset_data_num()
        
    @contextmanager
    def Lock(self):
        """전체 LockManager에 대한 상위 락 획득"""
        
        self.OverAllLock.acquire()
        try:
            yield
        finally:
            self.OverAllLock.release()

    @contextmanager
    def acquire_next_available_lock_and_shm(self, retry_interval=0.5):
        """
        Round Robin 방식으로 락을 획득하며, 락을 획득할 때까지 무한 재시도.
        :param retry_interval: 재시도 간격 (초)
        """
        
        total_locks = len(self.shm_mutexes)

        while True:  # 무한 재시도
            acquired_mutex_shm = None

            # Round Robin 방식으로 락 순회
            for _ in range(total_locks):
                with self.current_index.get_lock():
                    index = self.current_index.value
                    self.current_index.value = (index + 1) % total_locks

                lock, shm_interface = self.shm_mutexes[index]
                if lock.acquire(block=False):  # 논블락 락 획득 시도
                    acquired_mutex_shm = (lock, shm_interface)
                    break

            if acquired_mutex_shm is not None:
                lock, shm_interface = acquired_mutex_shm
                try:
                    yield shm_interface  # 공유 메모리 인터페이스 반환
                finally:
                    lock.release()  # 락 해제
                return  # 성공적으로 락을 반환했으므로 종료

            # 락 획득 실패 시 대기 후 재시도
            time.sleep(retry_interval)