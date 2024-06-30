import os, sys
# import io
import signal
# import atexit
import time
# import gymnasium as gym
# import copy

import torch
import traceback
import asyncio
import multiprocessing as mp

from types import SimpleNamespace as SN
from pathlib import Path
from multiprocessing import Process
# from datetime import datetime

from agents.storage_module.shared_batch import (
    setup_shared_memory,
)
from agents.learner import (
    LearnerSinglePPO,
    LearnerSingleIMPALA,
)
from agents.worker import Worker, TestWorker
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
    ErrorComment,
    select_least_used_gpu,
)
from utils.lock import Mutex


fn_dict = {}
child_process = {} # 전역 변수로 child_process 관리
terminate_flag = False


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
            map_name=self.args.map_name,
            debug=True,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=False,
            min_attack_range=2,
            fully_observable=True,
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
        self.ModelCls = module_name_space.model_cls
    
    @staticmethod
    def extract_err(target_dir: str):
        log_dir = os.path.join(os.getcwd(), "logs", target_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        return traceback.format_exc(limit=128), log_dir

    @staticmethod
    def worker_run(
        model_cls, worker_name, stop_event, args, manager_ip, learner_ip, port, learner_port, env_space, heartbeat=None
    ):
        worker = Worker(
            args, model_cls, worker_name, stop_event, manager_ip, learner_ip, port, learner_port, env_space, heartbeat
        )
        asyncio.run(worker.life_cycle_chain())  # collect rollout

    @staticmethod
    def test_worker_run(
        model_cls, worker_name, stop_event, args, manager_ip, learner_ip, port, learner_port, env_space, heartbeat=None
    ):
        test_worker = TestWorker(
            args, model_cls, worker_name, stop_event, manager_ip, learner_ip, port, learner_port, env_space, heartbeat
        )
        test_worker.render_test()

    @staticmethod
    def storage_run(
        args,
        mutex,
        shm_ref,
        stop_event,
        learner_ip,
        learner_port,
        env_space,
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
            heartbeat,
        )
        asyncio.run(storage.shared_memory_chain())

    @staticmethod
    def learner_run(
        model_cls,
        learner_cls,
        args,
        mutex,
        shm_ref,
        stop_event,
        learner_ip,
        learner_port,
        env_space,
        heartbeat=None,
    ):
        learner = learner_cls(
            args,
            mutex,
            model_cls,
            shm_ref,
            stop_event,
            learner_ip,
            learner_port,
            env_space,
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
        manager = Manager(
            self.args, self.stop_event, manager_ip, learner_ip, port, learner_port
        )
        asyncio.run(manager.data_chain())

    @register
    def worker_sub_process(self, num_p, manager_ip, learner_ip, port, learner_port):
        for i in range(int(num_p)):
            print("Build Worker {:d}".format(i))
            worker_name = "worker_" + str(i)
            
            heartbeat = mp.Value("f", time.monotonic())

            src_w = {
                "target": Runner.worker_run,
                "args": (
                    self.ModelCls,
                    worker_name,
                    self.stop_event,
                    self.args,
                    manager_ip, 
                    learner_ip,
                    port,
                    learner_port,
                    self.env_space,
                ),
                "kwargs": {"heartbeat": heartbeat},
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

        # for wp in child_process:
        #     wp.join()

    @register
    def learner_sub_process(self, learner_ip, learner_port):
        mutex = Mutex()

        # 학습을 위한 공유메모리 확보
        shm_ref = setup_shared_memory(self.env_space)

        heartbeat = mp.Value("f", time.monotonic())

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
                "heartbeat": heartbeat,
            },
        }

        s = Process(
            target=src_s.get("target"),
            args=src_s.get("args"),
            kwargs=src_s.get("kwargs"),
            daemon=True,
        )  # child-processes
        child_process.update({s: src_s})

        heartbeat = mp.Value("f", time.monotonic())

        src_l = {
            "target": Runner.learner_run,
            "args": (
                self.ModelCls,
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
                "heartbeat": heartbeat,
            },
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

        # for lp in child_process:
        #     lp.join()

    @register
    def worker_test_process(self, num_p, manager_ip, learner_ip, port, learner_port):
        """TestWorker를 (학습) Worker를 상속받아 구현.
        num_p, manager_ip, learner_ip, port, learner_port는 불필요한 dummy args
        """
        
        print("Build Test Worker")
        worker_name = "test worker"
        
        Runner.test_worker_run(
            self.ModelCls,
            worker_name,
            self.stop_event,
            self.args,
            manager_ip, 
            learner_ip,
            port,
            learner_port,
            self.env_space,
        )

    def start(self):
        def _monitor_child_process(restart_delay=30):
            def _restart_process(src, heartbeat):
                traceback.print_exc(limit=128)

                err, log_dir = Runner.extract_err(func_name)
                SaveErrorLog(err, log_dir)
                
                # heartbeat 기록 갱신
                heartbeat.value = time.monotonic()

                new_p = Process(
                    target=src.get("target"),
                    args=src.get("args"),
                    kwargs=src.get("kwargs"),
                    daemon=True,
                )  # child-processes
                new_p.start()
                
                child_process.update({new_p: src})

            def _signal_handler(signum, frame):
                global terminate_flag
                terminate_flag = True
                # 자식 프로세스 종료 신호 보냄
                self.stop_event.set()
                for p in list(child_process.keys()):
                    p.terminate()
                    p.join()
                sys.exit(0)

            # SIGINT 시그널 핸들러 등록 (ctrl + c 감지)
            signal.signal(signal.SIGINT, _signal_handler)

            while not terminate_flag:
                for p in list(child_process.keys()):
                    src = child_process.get(p)
 
                    heartbeat = src["kwargs"].get("heartbeat")
                    assert heartbeat is not None

                    # 자식 프로세스가 죽었거나, 일정 시간 이상 통신이 안된 경우 -> 재시작
                    if not p.is_alive() or (
                        (time.monotonic() - heartbeat.value) > restart_delay
                    ):
                        # 해당 자식 프로세스 종료
                        p.terminate()
                        p.join()
                        child_process.pop(p)
                        assert not p.is_alive(), f"p: {p}"

                        # 해당 자식 프로세스 신규 생성 및 시작
                        _restart_process(src, heartbeat)

                time.sleep(1.0)
        
        assert len(sys.argv) >= 1
        func_name = sys.argv[1]
        func_args = sys.argv[2:]
        
        assert func_name in ("worker_sub_process", "manager_sub_process", "learner_sub_process", "worker_test_process")
        
        # # 자식 프로세스 종료 함수
        # def terminate_processes(processes):
        #     for p in processes:
        #         if p.is_alive():
        #             p.terminate()
        #             p.join()

        # # 종료 시그널 핸들러 설정
        # def signal_handler(signum, frame):
        #     print("Signal received, terminating processes")
        #     terminate_processes(child_process)

        # signal.signal(signal.SIGINT, signal_handler)
        # signal.signal(signal.SIGTERM, signal_handler)

        # # 프로세스 종료 시 실행될 함수 등록
        # atexit.register(terminate_processes, child_process)
        
        try:
            if func_name in fn_dict:
                fn_dict[func_name](self, *func_args)
            else:
                assert False, f"Wronf func_name: {func_name}, func_args: {func_args}"
            _monitor_child_process()
            
        finally:
            traceback.print_exc(limit=128)

            err, log_dir = Runner.extract_err(func_name)
            SaveErrorLog(err, log_dir)
            
            KillProcesses(os.getpid())


if __name__ == "__main__":
    rn = Runner()
    rn.start()
