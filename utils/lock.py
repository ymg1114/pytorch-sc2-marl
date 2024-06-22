from contextlib import contextmanager

import queue
import torch.multiprocessing as mp


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
        return

    def get(self, q: mp.Queue):
        with self.lock():
            data = q.get()
        return data

    def put(self, q: mp.Queue, data):
        with self.lock():
            q.put(data)
        return


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
        return

    def get(self, q: mp.Queue):
        with self.lock():
            data = q.get()
        return data

    def get_with_timeout(self, q: mp.Queue, timeout=1.0):
        with self.lock():
            try:
                data = q.get(timeout=timeout)  # timeout초 동안 대기
            except queue.Empty:
                data = None
                print(f"Queue is empty after waiting for {timeout} seconds")
        return data

    def put(self, q: mp.Queue, data):
        with self.lock():
            q.put(data)
        return
