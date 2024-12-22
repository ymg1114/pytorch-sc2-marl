import os, sys
# import io
import jax
import ctypes
import json
import torch
import time
import platform
import psutil
import pickle
import blosc2
import subprocess
import numpy as np

from functools import reduce
import operator

from collections import deque, defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path

from datetime import datetime
from signal import SIGTERM  # or SIGKILL
from types import SimpleNamespace


NO_OP_IDX = 0
STOP_IDX = 1
MOVE_IDX = 2
TARGET_IDX = 3
FLEE_IDX = 4

MOVE_NORTH_IDX = 2
MOVE_SOUTH_IDX = 3
MOVE_EAST_IDX = 4
MOVE_WEST_IDX = 5

ACT = [NO_OP_IDX, STOP_IDX, MOVE_IDX, TARGET_IDX, FLEE_IDX ] # no-op, stop, move, target, flee
MOVE = [MOVE_NORTH_IDX, MOVE_SOUTH_IDX, MOVE_EAST_IDX, MOVE_WEST_IDX]


# Determine the library name based on the platform
if sys.platform.startswith("win"):
    lib_name = Path("observer") / "cxx_flee_algo" / "lib_win" / "FleeAlgorithm.dll"
else:
    assert sys.platform == "linux", f"running platform is '{sys.platform}'"
    lib_name = Path("observer") / "cxx_flee_algo" / "lib_linux" / "libFleeAlgorithm.so"

# Load the shared library
lib = ctypes.CDLL(str(lib_name))

# Define Position struct
class Position(ctypes.Structure):
    _fields_ = [("y", ctypes.c_int), ("x", ctypes.c_int)]

# Define the function prototype
lib.compute_flee_positions.argtypes = [
    ctypes.c_int,  # n_ally
    ctypes.c_int,  # n_enemy
    ctypes.c_int,  # n_rows
    ctypes.c_int,  # n_cols
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # grid_maps
    ctypes.POINTER(ctypes.c_bool),  # alive_allies
    ctypes.POINTER(ctypes.c_bool),  # alive_enemies
    ctypes.POINTER(Position),  # positions_allies
    ctypes.POINTER(Position),  # positions_enemies
    ctypes.c_int,  # min_search_length
    ctypes.c_int,  # max_search_length
    ctypes.c_float,  # ally_weight
    ctypes.c_float,  # enemy_weight
    ctypes.POINTER(Position),  # flee_positions (output)
]


def select_least_used_jax_gpu():
    # GPU 메모리 정보를 가져옴
    gpu_memory_info = get_gpu_memory_info()

    # 가장 Free Memory (MiB)가 높은 GPU 선택
    least_used_idx = max(enumerate(gpu_memory_info), key=lambda x: x[1]['Free Memory (MiB)'])[0]
    
    gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpu_devices:
        return None
    
    return gpu_devices[least_used_idx]


def dict_to_simplenamespace(d):
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_simplenamespace(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_simplenamespace(item) for item in d]
    else:
        return d


utils = os.path.join(os.getcwd(), "utils", "parameters.json")
with open(utils) as f:
    _p = json.load(f)
    Params = SimpleNamespace(**_p)


utils = os.path.join(os.getcwd(), "utils", "machines.json")
with open(utils) as f:
    _p = json.load(f)
    Machines = dict_to_simplenamespace(_p)


utils = os.path.join(os.getcwd(), "utils", "sc2_config", f"sc2_{Params.map_name}_config.json")
with open(utils) as f:
    SC2Config = json.load(f)


class SingletonMetaCls(type):
    __instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super().__call__(*args, **kwargs)
        return cls.__instances[cls]


dt_string = datetime.now().strftime(f"[%d][%m][%Y]-%H_%M")
result_dir = os.path.join("results", str(dt_string))
model_dir = os.path.join(result_dir, "models")


ErrorComment = "Should be PPO or IMPALA"


flatten = lambda obj: obj.numpy().reshape(-1).astype(np.float32)


def to_torch(array):
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).float()
    elif isinstance(array, torch.Tensor):
        return array
    else:
        raise TypeError("Input should be a numpy array or a torch tensor")
    

def extract_file_num(filename):
    parts = filename.stem.split("_")
    try:
        return int(parts[-1])
    except ValueError:
        return -1


# def make_gpu_batch(*args, device):
#     to_gpu = lambda tensor: tensor.to(device)
#     return tuple(map(to_gpu, args))


def get_gpu_memory_info():
    # nvidia-smi 명령어를 실행하여 출력 값을 받아옴
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip()

    # 각 줄을 파싱하여 메모리 정보 추출
    gpu_info = output.split('\n')
    gpu_memory_info = []
    for i, info in enumerate(gpu_info):
        total, used, free = map(int, info.split(', '))
        gpu_memory_info.append({
            'GPU': i,
            'Total Memory (MiB)': total,
            'Used Memory (MiB)': used,
            'Free Memory (MiB)': free
        })

    return gpu_memory_info

def select_least_used_gpu():
    # GPU 메모리 정보를 가져옴
    gpu_memory_info = get_gpu_memory_info()

    # 가장 Free Memory (MiB)가 높은 GPU 선택
    best_gpu = max(gpu_memory_info, key=lambda x: x['Free Memory (MiB)'])
    return best_gpu['GPU']


def get_current_process_id():
    return os.getpid()


def get_process_id(name):
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["name"] == name:
            return proc.info["pid"]


def kill_process(pid):
    os.kill(pid, SIGTERM)


class KillSubProcesses:
    def __init__(self, processes):
        self.target_processes = processes
        self.os_system = platform.system()

    def __call__(self):
        assert hasattr(self, "target_processes")
        for p in self.target_processes:
            if self.os_system == "Windows":
                print("This is a Windows operating system.")
                p.terminate()  # Windows에서는 "관리자 권한" 이 필요.

            elif self.os_system == "Linux":
                print("This is a Linux operating system.")
                os.kill(p.pid, SIGTERM)


def mul(shape_dim):
    return reduce(operator.mul, shape_dim, 1)


def extract_values(list_of_dicts, key):
    return [d[key] for d in list_of_dicts]


def extract_nested_values(deque_of_dicts, outer_key, inner_key):
    return [d[outer_key][inner_key] for d in deque_of_dicts]


def counted(f):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return f(*args, **kwargs)

    wrapper.calls = 0
    return wrapper


class ExecutionTimer:
    def __init__(self, threshold=100, num_transition=None):
        self.num_transition = num_transition
        self.timer_dict = defaultdict(lambda: deque(maxlen=threshold))
        self.throughput_dict = defaultdict(lambda: deque(maxlen=threshold))

    @contextmanager
    def timer(self, code_block_name: str, check_throughput=False):
        start_time = time.time()
        yield  # 사용자가 지정 코드 블록이 실행되는 부분
        end_time = time.time()

        elapsed_time = end_time - start_time

        self.timer_dict[code_block_name].append(elapsed_time)  # sec
        if self.num_transition is not None and isinstance(
            self.num_transition, (int, float, np.number)
        ):
            if check_throughput is True:
                self.throughput_dict[code_block_name].append(
                    self.num_transition / (elapsed_time + 1e-6)
                )  # transition/sec
        # avg_time = sum(self.exec_times) / len(self.exec_times)


def SaveErrorLog(error: str, log_dir: str):
    current_time = time.strftime("[%Y_%m_%d][%H_%M_%S]", time.localtime(time.time()))
    # log_dst = os.path.join(log_dir, f"error_log_{current_time}.txt")
    dir = Path(log_dir)
    error_log = dir / f"error_log_{current_time}.txt"
    error_log.write_text(f"{error}\n")


class Protocol(Enum):
    Model = auto()
    Rollout = auto()
    Stat = auto()


def KillProcesses(pid):
    parent = psutil.Process(pid)
    for child in parent.children(
        recursive=True
    ):  # or parent.children() for recursive=False
        child.kill()
    parent.kill()


def encode(protocol, data):
    return pickle.dumps(protocol), blosc2.compress(pickle.dumps(data), clevel=1)


def decode(protocol, data):
    return pickle.loads(protocol), pickle.loads(blosc2.decompress(data))


if __name__ == "__main__":
    KillProcesses()
