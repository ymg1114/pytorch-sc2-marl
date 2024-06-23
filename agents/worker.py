import uuid
import time
import zmq
import asyncio
import torch
# import torch.jit as jit
import numpy as np

from utils.utils import Protocol, encode, decode, SC2Config

from env.sc2_env_wrapper import WrapperSMAC2

from rewarder.rewarder import REWARD_PARAM

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from networks.network import ModelSingle


class Worker:
    def __init__(
        self, args, model, worker_name, stop_event, manager_ip, learner_ip, port, learner_port, heartbeat=None
    ):
        self.args = args
        self.device = args.device  # cpu

        self.model: "ModelSingle" = model.to(torch.device("cpu")).eval()
        self.worker_name = worker_name
        self.stop_event = stop_event
        self.heartbeat = heartbeat

        self.env = WrapperSMAC2(
            capability_config=SC2Config,
            map_name=args.map_name,
            debug=True,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
        )

        self.env_info = self.env.get_env_info()
        self.env_space = self.env.get_env_space(self.args)

        self.zeromq_set(manager_ip, learner_ip, port, learner_port)

    def __del__(self):  # 소멸자
        if hasattr(self, "pub_socket"):
            self.pub_socket.close()
        if hasattr(self, "sub_socket"):
            self.sub_socket.close()

    def zeromq_set(self, manager_ip, learner_ip, port, learner_port):
        context = zmq.asyncio.Context()

        # worker <-> manager
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{manager_ip}:{port}")  # publish rollout, stat

        self.sub_socket = context.socket(zmq.SUB)
        self.sub_socket.connect(
            f"tcp://{learner_ip}:{int(learner_port) + 1}"
        )  # subscribe model
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

    async def req_model(self):
        while not self.stop_event.is_set():
            protocol, data = decode(*await self.sub_socket.recv_multipart())
            if protocol is Protocol.Model:
                model_state_dict = {k: v.to("cpu") for k, v in data.items()}
                self.model.load_state_dict(model_state_dict)  # reload learned-model from learner

            await asyncio.sleep(0.1)

    async def pub_rollout(self, **roll_out):
        await self.pub_socket.send_multipart([*encode(Protocol.Rollout, roll_out)])

    async def pub_stat(self, info):
        stat_dict = {
            "info": info,
            "epi_rew_vec": self.epi_rew_vec
        }
        await self.pub_socket.send_multipart([*encode(Protocol.Stat, stat_dict)])
        print(
            f"worker_name: {self.worker_name} epi_rew_vec: {self.epi_rew_vec} pub stat to manager!"
        )

    async def life_cycle_chain(self):
        tasks = [
            asyncio.create_task(self.collect_rolloutdata()),
            asyncio.create_task(self.req_model()),
        ]
        await asyncio.gather(*tasks)

    def initialize_lstm(self):
        return (
            torch.zeros(self.env_info["n_agents"], self.args.hidden_size),
            torch.zeros(self.env_info["n_agents"], self.args.hidden_size),
        )

    def set_default_actions(self, act_dict):
        act_dict["on_select_act"] = torch.ones(self.env_info["n_agents"], 1)  # 항상 선택
        act_dict["on_select_move"] = torch.zeros(self.env_info["n_agents"], 1)
        act_dict["on_select_target"] = torch.zeros(self.env_info["n_agents"], 1)

    def create_rollout_dict(self, _id, obs_dict, act_dict, rew_vec, is_first, done_vec, dead_agents_vec):
        return {
            "id": [f"{_id}_{agent_id}" for agent_id in range(self.env_info["n_agents"])],
            "obs_dict": {k: torch.from_numpy(v) for k, v in obs_dict.items()},
            "act_dict": act_dict,
            "rew_vec": torch.from_numpy(rew_vec),
            "is_fir": torch.ones(self.env_info["n_agents"]) if is_first else torch.zeros(self.env_info["n_agents"]),
            "done": (done_vec.to(torch.bool) | dead_agents_vec.to(torch.bool)).to(done_vec.dtype),  # 경기 종료 혹은 죽은 에이전트는 done 처리
        }

    async def collect_rolloutdata(self):
        print(f"Build Environment for {self.worker_name}")

        while not self.stop_event.is_set():
            self.env.reset()
            _id = str(uuid.uuid4()) # 각 경기의 고유한 난수
            
            hx, cx = self.initialize_lstm()
            self.epi_rew_vec = np.zeros((self.env_info["n_agents"], len(REWARD_PARAM)), dtype=np.float32)
            dead_agents_vec = torch.zeros(self.env_info["n_agents"])
            
            is_first = True
            for _ in range(self.env_info["episode_limit"]):
                obs_dict = self.env.get_obs_dict()
                act_dict = self.model.act(obs_dict, hx, cx)
                self.set_default_actions(act_dict)
                
                rew_vec, terminated, info = self.env.step_dict(act_dict, dead_agents_vec)
                # self.env.render() # Uncomment for rendering
                
                self.epi_rew_vec += rew_vec
                done_vec = torch.ones(self.env_info["n_agents"]) if terminated else torch.zeros(self.env_info["n_agents"])

                roll_out = self.create_rollout_dict(_id, obs_dict, act_dict, rew_vec, is_first, done_vec, dead_agents_vec)
                await self.pub_rollout(**roll_out)

                is_first = False
                hx = act_dict["hx"]
                cx = act_dict["cx"]
                
                await asyncio.sleep(0.15)

                if self.heartbeat is not None:
                    self.heartbeat.value = time.time()

                if terminated:
                    # if not info.get("battle_won", True):
                    #     assert bool(dead_agents_vec.all()) # 패배한 경우, 모든 에이전트는 반드시 사망해야 함
                    break

            await self.pub_stat(info) # 경기 종료 시 반환하는 info 포함