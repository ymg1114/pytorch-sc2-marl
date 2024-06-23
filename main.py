import os, sys
import signal
import atexit
import time
# import gymnasium as gym
import copy

import torch
import traceback
import asyncio
import multiprocessing as mp

from types import SimpleNamespace as SN
from pathlib import Path
from multiprocessing import Process
from datetime import datetime

from agents.storage_module.shared_batch import (
    setup_shared_memory,
)
from agents.learner import (
    LearnerSinglePPO,
    LearnerSingleIMPALA,
)
from agents.worker import Worker
from agents.learner_storage import LearnerStorage
from agents.manager import Manager

from networks.network import (
    ModelSingle,
)
from env.sc2_env_wrapper import WrapperSMAC2

from utils.utils import (
    KillProcesses,
    SaveErrorLog,
    Params,
    SC2Config,
    result_dir,
    model_dir,
    extract_file_num,
    ErrorComment,
    select_least_used_gpu,
)
from utils.lock import Mutex


fn_dict = {}
child_process = {}


def register(fn):
    fn_dict[fn.__name__] = fn
    return fn


class Runner:
    def __init__(self):
        mp.set_start_method('spawn')
        self.args = Params
        self.args.device = torch.device(
            f"cuda:{select_least_used_gpu()}" if torch.cuda.is_available() else "cpu"
        )

        # 미리 정해진 경로가 있다면 그것을 사용
        self.args.result_dir = self.args.result_dir or result_dir
        self.args.model_dir = self.args.model_dir or model_dir
        print(f"device: {self.args.device}")

        _model_dir = Path(self.args.model_dir)
        # 경로가 디렉토리인지 확인
        if not _model_dir.is_dir():
            # 디렉토리가 아니라면 디렉토리 생성
            _model_dir.mkdir(parents=True, exist_ok=True)

        env = WrapperSMAC2(
            capability_config=SC2Config,
            map_name="10gen_terran",
            debug=True,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
        )
        self.env_info = env.get_env_info()
        self.env_space = env.get_env_space(self.args)
        env.close()
        
        self.stop_event = mp.Event()

        module_switcher = {  # (learner_cls, model_cls)
            "PPO": SN(learner_cls=LearnerSinglePPO, model_cls=ModelSingle),
            "IMPALA": SN(learner_cls=LearnerSingleIMPALA, model_cls=ModelSingle),
        }
        module_name_space = module_switcher.get(
            self.args.algo, lambda: AssertionError(ErrorComment)
        )

        self.LearnerCls = module_name_space.learner_cls

        self.Model = module_name_space.model_cls(self.args, self.env_space)
        self.Model.to(torch.device("cpu"))  # cpu 모델

    def set_model_weight(self, model_dir):
        model_files = list(Path(model_dir).glob(f"{self.args.algo}_*.pt"))

        prev_model_weight = None
        if len(model_files) > 0:
            sorted_files = sorted(model_files, key=extract_file_num)
            if sorted_files:
                prev_model_weight = torch.load(
                    sorted_files[-1],
                    map_location=torch.device("cpu"),  # 가장 최신 학습 모델 로드
                )

        learner_model_state_dict = self.Model.cpu().state_dict()
        if prev_model_weight is not None:
            learner_model_state_dict = {
                k: v.cpu() for k, v in prev_model_weight.state_dict().items()
            }

        return learner_model_state_dict  # cpu 텐서

    @staticmethod
    def extract_err(target_dir: str):
        log_dir = os.path.join(os.getcwd(), "logs", target_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        return traceback.format_exc(limit=128), log_dir

    @staticmethod
    def worker_run(
        model, worker_name, stop_event, args, manager_ip, learner_ip, port, learner_port, heartbeat=None
    ):
        worker = Worker(
            args, model, worker_name, stop_event, manager_ip, learner_ip, port, learner_port, heartbeat
        )
        asyncio.run(worker.life_cycle_chain())  # collect rollout

    @staticmethod
    def storage_run(
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
        storage = LearnerStorage(
            args,
            mutex,
            shm_ref,
            stop_event,
            learner_ip,
            learner_port,
            env_space,
            shared_stat_array,
            heartbeat,
        )
        asyncio.run(storage.shared_memory_chain())

    @staticmethod
    def run_learner(
        learner_model,
        learner_cls,
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
        learner = learner_cls(
            args,
            mutex,
            learner_model,
            shm_ref,
            stop_event,
            learner_ip,
            learner_port,
            env_space,
            shared_stat_array,
            heartbeat,
        )
        learning_chain_switcher = {
            "PPO": learner.learning_chain_ppo,
            "IMPALA": learner.learning_chain_impala,
        }
        learning_chain = learning_chain_switcher.get(
            args.algo, lambda: AssertionError(ErrorComment)
        )
        asyncio.run(learning_chain())

    @register
    def manager_sub_process(self, manager_ip, learner_ip, port, learner_port):
        try:
            manager = Manager(
                self.args, self.stop_event, manager_ip, learner_ip, port, learner_port
            )
            asyncio.run(manager.data_chain())
        except:
            # 자식 프로세스 종료 신호 보냄
            self.stop_event.set()

            traceback.print_exc(limit=128)

            err, log_dir = Runner.extract_err("manager")
            SaveErrorLog(err, log_dir)

    @register
    def worker_sub_process(self, num_p, manager_ip, learner_ip, port, learner_port):
        try:
            learner_model_state_dict = self.set_model_weight(self.args.model_dir)

            for i in range(int(num_p)):
                print("Build Worker {:d}".format(i))
                worker_model = copy.deepcopy(self.Model)
                worker_model.load_state_dict(learner_model_state_dict)
                worker_name = "worker_" + str(i)

                heartbeat = mp.Value("f", time.time())

                src_w = {
                    "target": Runner.worker_run,
                    "args": (
                        worker_model,
                        worker_name,
                        self.stop_event,
                        self.args,
                        manager_ip, 
                        learner_ip,
                        port,
                        learner_port,
                    ),
                    "kwargs": {"heartbeat": heartbeat},
                    "heartbeat": heartbeat,
                    "is_model_reload": True,
                }

                w = Process(
                    target=src_w.get("target"),
                    args=src_w.get("args"),
                    kwargs=src_w.get("kwargs"),
                    daemon=True,
                )  # child-processes
                child_process.update({w: src_w})

            for wp in child_process:
                wp.start()

            for wp in child_process:
                wp.join()

        except:
            # 자식 프로세스 종료 신호 보냄
            self.stop_event.set()

            traceback.print_exc(limit=128)

            err, log_dir = Runner.extract_err("worker")
            SaveErrorLog(err, log_dir)

            for wp in child_process:
                wp.terminate()

    @register
    def learner_sub_process(self, learner_ip, learner_port):
        try:
            learner_model_state_dict = self.set_model_weight(self.args.model_dir)
            self.Model.load_state_dict(learner_model_state_dict)

            mutex = Mutex()

            # 학습을 위한 공유메모리 확보
            shm_ref = setup_shared_memory(self.env_space)
            
            len_rew_vec = self.env_space["rew"]["rew_vec"].nvec[-1]
            # TODO: 좋은 구조는 아님.
            shared_stat_array = mp.Array(
                "f", 5+int(len_rew_vec)
            )  # [global game counts, activate, rew_vec~]

            heartbeat = mp.Value("f", time.time())

            src_s = {
                "target": Runner.storage_run,
                "args": (
                    self.args,
                    mutex,
                    shm_ref,
                    self.stop_event,
                    learner_ip,
                    learner_port,
                    self.env_space,
                ),
                "kwargs": {
                    "shared_stat_array": shared_stat_array,
                    "heartbeat": heartbeat,
                },
                "heartbeat": heartbeat,
            }

            s = Process(
                target=src_s.get("target"),
                args=src_s.get("args"),
                kwargs=src_s.get("kwargs"),
                daemon=True,
            )  # child-processes
            child_process.update({s: src_s})

            heartbeat = mp.Value("f", time.time())

            src_l = {
                "target": Runner.run_learner,
                "args": (
                    self.Model,
                    self.LearnerCls,
                    self.args,
                    mutex,
                    shm_ref,
                    self.stop_event,
                    learner_ip,
                    learner_port,
                    self.env_space,
                ),
                "kwargs": {
                    "shared_stat_array": shared_stat_array,
                    "heartbeat": heartbeat,
                },
                "heartbeat": heartbeat,
                "is_model_reload": True,
            }

            l = Process(
                target=src_l.get("target"),
                args=src_l.get("args"),
                kwargs=src_l.get("kwargs"),
                daemon=True,
            )  # child-processes
            child_process.update({l: src_l})

            for lp in child_process:
                lp.start()

            for lp in child_process:
                lp.join()

        except:
            # 자식 프로세스 종료 신호 보냄
            self.stop_event.set()

            traceback.print_exc(limit=128)

            err, log_dir = Runner.extract_err("learner")
            SaveErrorLog(err, log_dir)

            for lp in child_process:
                lp.terminate()

    def start(self):
        assert len(sys.argv) >= 1
        func_name = sys.argv[1]
        func_args = sys.argv[2:]
        
        if (
            func_name != "manager_sub_process"
        ):  # manager는 관리할 자식 프로세스가 없기 때문.
            assert func_name in ("worker_sub_process", "learner_sub_process")

            # 자식 프로세스 종료 함수
            def terminate_processes(processes):
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                        p.join()

            # 종료 시그널 핸들러 설정
            def signal_handler(signum, frame):
                print("Signal received, terminating processes")
                terminate_processes(child_process)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # 프로세스 종료 시 실행될 함수 등록
            atexit.register(terminate_processes, child_process)

        try:
            if func_name in fn_dict:
                fn_dict[func_name](self, *func_args)
            else:
                assert False, f"Wronf func_name: {func_name}, func_args: {func_args}"

        except Exception as e:
            # 자식 프로세스 종료 신호 보냄
            self.stop_event.set()

            print(f"error: {e}")
            traceback.print_exc(limit=128)

            for p in child_process:
                if p.is_alive():
                    p.terminate()
                    p.join()

        finally:
            KillProcesses(os.getpid())


if __name__ == "__main__":
    rn = Runner()
    rn.start()
