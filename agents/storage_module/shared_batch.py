import numpy as np
import multiprocessing as mp

from utils.utils import mul


def set_shared_memory(shm_ref, key, ndarray):
    """멀티프로세싱 환경에서 데이터 복사 없이 공유 메모리를 통해 데이터를 공유함으로써 성능을 개선할 수 있음."""

    shm_ref.update(
        {key: (mp.Array("f", len(ndarray)), np.float32)}
    )  # {키워드: (공유메모리, 타입), ... }


def initialize_shared_memory(shm_ref, space):
    for k, v in space.items():
        set_shared_memory(shm_ref, k, np.zeros(mul(v.nvec), dtype=np.float32))


def setup_shared_memory(env_space):
    shm_ref = {}  # Learner / LearnerStorage 에서 공유할 메모리 주소를 담음

    initialize_shared_memory(shm_ref, env_space["obs"])
    initialize_shared_memory(shm_ref, env_space["act"])
    initialize_shared_memory(shm_ref, env_space["rew"])
    initialize_shared_memory(shm_ref, env_space["info"])

    # 공유메모리 저장 인덱스
    shm_ref["batch_index"] = mp.Value("i", 0, lock=True) # 0 값 인덱스로 초기화
    return shm_ref


class SMInterface:
    def __init__(self, shm_ref, env_space):
        self.shm_ref = shm_ref
        self.env_space = env_space
        self.shared_memory_spaces = list(env_space.keys()) # ["obs", "act", "rew", "info"]
        self.sh_data_num: mp.Value = self.shm_ref.get("batch_index")
        
    def shm_ndarray_interface(self, name: str):
        assert name in self.shm_ref, f"{name} not found in shared memory reference"
        shm_memory_tuple = self.shm_ref.get(name)
        
        assert shm_memory_tuple is not None, f"Shared memory tuple for {name} is None"
        shm_array, dtype = shm_memory_tuple
        
        # 공유 메모리 객체에서 실제 버퍼를 가져와, ndarray 배열로 변환 (생성)
        return np.frombuffer(buffer=shm_array.get_obj(), dtype=dtype, count=-1)

    def get_shared_memory_interface(self):
        for space in self.shared_memory_spaces:
            for k in self.env_space[space].keys():
                setattr(self, f"sh_{k}", self.shm_ndarray_interface(k))
                
        self.reset_data_num() # 공유메모리 저장 인덱스 (batch_num) 초기화

    def reset_data_num(self):
        with self.sh_data_num.get_lock(): # 락을 사용해 동기화
            self.sh_data_num.value = 0