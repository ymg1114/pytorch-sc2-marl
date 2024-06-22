import zmq.asyncio

import numpy as np
import multiprocessing as mp

from utils.utils import mul


def set_shared_memory(shm_ref, name, ndarray):
    """멀티프로세싱 환경에서 데이터 복사 없이 공유 메모리를 통해 데이터를 공유함으로써 성능을 개선할 수 있음."""

    shm_ref.update(
        {name: (mp.Array("f", len(ndarray)), np.float32)}
    )  # {키워드: (공유메모리, 타입), ... }


def setup_shared_memory(env_space):
    shm_ref = {}  # Learner / LearnerStorage 에서 공유할 메모리 주소를 담음
    
    obs_space = env_space["obs"]
    act_space = env_space["act"]
    rew_space = env_space["rew"]
    info_space = env_space["info"]
    
    for k, v in obs_space.items():
        set_shared_memory(shm_ref, k, np.zeros(mul(v.nvec), dtype=np.float32))
        
    for k, v in act_space.items():
        set_shared_memory(shm_ref, k, np.zeros(mul(v.nvec), dtype=np.float32))
        
    for k, v in rew_space.items():
        set_shared_memory(shm_ref, k, np.zeros(mul(v.nvec), dtype=np.float32))
        
    for k, v in info_space.items():
        set_shared_memory(shm_ref, k, np.zeros(mul(v.nvec), dtype=np.float32))
        
    # 공유메모리 저장 인덱스
    sh_data_num = mp.Value("i", 0)
    sh_data_num.value = 0  # 초기화
    shm_ref.update({"batch_index": sh_data_num})
    return shm_ref


def reset_shared_on_policy_memory(env_space):
    return setup_shared_memory(env_space)


class SMInterFace:
    def __init__(self, shm_ref, env_space):
        self.shm_ref = shm_ref
        self.env_space = env_space

    def shm_ndarry_inferface(self, name: str):
        assert hasattr(self, "shm_ref") and name in self.shm_ref

        shm_memory_tuple = self.shm_ref.get(name)
        assert shm_memory_tuple is not None

        shm_array = shm_memory_tuple[0]
        dtype = shm_memory_tuple[1]

        return np.frombuffer(buffer=shm_array.get_obj(), dtype=dtype, count=-1)

    def get_shared_memory_interface(self):
        assert hasattr(self, "shm_ref")
        
        obs_space = self.env_space["obs"]
        act_space = self.env_space["act"]
        rew_space = self.env_space["rew"]
        info_space = self.env_space["info"]
        
        for k in obs_space.keys():
            setattr(self, f"sh_{k}", self.shm_ndarry_inferface(k))
                
        for k in act_space.keys():
            setattr(self, f"sh_{k}", self.shm_ndarry_inferface(k))
            
        for k in rew_space.keys():
            setattr(self, f"sh_{k}", self.shm_ndarry_inferface(k))
            
        for k in info_space.keys():
            setattr(self, f"sh_{k}", self.shm_ndarry_inferface(k))
                
        # self.sh_obs_batch = self.shm_ndarry_inferface("obs_batch")
        # self.sh_act_batch = self.shm_ndarry_inferface("act_batch")
        # self.sh_rew_batch = self.shm_ndarry_inferface("rew_batch")
        # self.sh_logits_batch = self.shm_ndarry_inferface("logits_batch")
        # self.sh_log_prob_batch = self.shm_ndarry_inferface("log_prob_batch")
        # self.sh_is_fir_batch = self.shm_ndarry_inferface("is_fir_batch")
        # self.sh_hx_batch = self.shm_ndarry_inferface("hx_batch")
        # self.sh_cx_batch = self.shm_ndarry_inferface("cx_batch")

        self.sh_data_num = self.shm_ref.get("batch_index")

        self.reset_data_num()  # 공유메모리 저장 인덱스 (batch_num) 초기화

    def reset_data_num(self):
        self.sh_data_num.value = 0
