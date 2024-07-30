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


def check_act_avail(env, act_dict, n_agents):
    converted_act_idx = env.act_dict_converter(act_dict)
    avail_act_list = []
    for a_id in range(n_agents):
        avail_acts = env.get_avail_agent_actions(a_id)
        avail_act_list.append(avail_acts)

    for c_a, av_l in zip(converted_act_idx, avail_act_list):
        assert av_l[c_a.item()] == 1, f"converted_act_idx: {converted_act_idx}, avail_act_list: {avail_act_list}"


# def Wrapper(func):
#     agent_tag = []
#     is_full = [False]
    
#     def _inner(agents, n_agents):
#         nonlocal agent_tag, is_full
#         func(agents, n_agents, agent_tag, is_full)
#     return _inner


# @Wrapper
def check_agent_id(agents, n_agents, agent_tag, is_full):
    if not is_full[0]:
        for agent in agents.values():
            agent_tag.append(agent.tag)
    
    if len(agent_tag) == n_agents:
        is_full[0] = True
    
    if is_full[0]:
        assert agent_tag == [v.tag for v in agents.values()], f"agent_tag: {agent_tag}, agents: {agents} "
    

class Worker:
    def __init__(
        self,
        args,
        model_cls,
        worker_name,
        stop_event,
        manager_ip,
        learner_ip,
        manager_port,
        learner_worker_port,
        env_space,
        heartbeat=None,
    ):
        self.args = args
        self.device = args.device  # cpu
        self.env_space = env_space
        
        # 서로 다른 worker 인스턴스들이 다른 weight로 초기화되나, learner에서 최초 1회 학습 후 
        # identical 한 weight를 얻으므로, 이 정도는 감수
        self.model: "ModelSingle" = model_cls(self.args, self.env_space).to("cpu").eval()
        out_dict  = model_cls.set_model_weight(self.args)
        if out_dict is not None:
            self.model.load_state_dict(out_dict["state_dict"])

        self.worker_name = worker_name
        self.stop_event = stop_event
        self.heartbeat = heartbeat

        self.env = WrapperSMAC2(
            capability_config=SC2Config,
            map_name=args.map_name,
            debug=True,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=False,
            min_attack_range=2,
            fully_observable=True,
        )

        self.env_info = self.env.get_env_info()
        self.env_space = self.env.get_env_space(self.args)

        self.zeromq_set(manager_ip, learner_ip, manager_port, learner_worker_port)

    def __del__(self):  # 소멸자
        if hasattr(self, "rollout_pub_socket"):
            self.rollout_pub_socket.close()
        if hasattr(self, "stat_pub_socket"):
            self.stat_pub_socket.close()
        if hasattr(self, "sub_socket"):
            self.sub_socket.close()
        if hasattr(self, "env"):
            self.env.close()
            
    def zeromq_set(self, manager_ip, learner_ip, manager_port, learner_worker_port):
        context = zmq.asyncio.Context()

        # worker <-> manager
        self.rollout_pub_socket = context.socket(zmq.PUB)
        self.rollout_pub_socket.connect(f"tcp://{manager_ip}:{manager_port}")  # publish rollout

        # worker <-> learner
        self.stat_pub_socket = context.socket(zmq.PUB)
        self.stat_pub_socket.connect(f"tcp://{learner_ip}:{int(learner_worker_port) + 2}")  # publish stat

        self.sub_socket = context.socket(zmq.SUB)
        self.sub_socket.connect(
            f"tcp://{learner_ip}:{int(learner_worker_port) + 1}"
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
        await self.rollout_pub_socket.send_multipart([*encode(Protocol.Rollout, roll_out)])

    async def pub_stat(self, info):
        stat_dict = {
            **info,
            **self.env.get_stats(),
            "epi_rew_vec": self.epi_rew_vec
        }
        await self.stat_pub_socket.send_multipart([*encode(Protocol.Stat, stat_dict)])
        print(
            f"worker_name: {self.worker_name} epi_rew_vec: {self.epi_rew_vec} pub stat to learner!"
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

    def create_rollout_dict_except_already_dead(self, _id, obs_dict, act_dict, rew_vec, is_first, done_vec, dead_agents_vec):
        rollout_dict = None

        death_tracker_ally = self.env.get_death_tracker_ally() # next-state에서 방금 막 죽은 것이 아닌, 이미 죽어 있는 아군 에이전트
        # tar_y = np.where(death_tracker_ally == 1)[0] # 이미 죽음
        not_tar_y = np.where(death_tracker_ally == 0)[0] # current-state 까진 살아 있음 혹은 next-state에서 막 방금 죽음

        id_v = [f"{_id}_{agent_id}" for agent_id in range(self.env_info["n_agents"]) if not death_tracker_ally[agent_id]]
        obs_dict_v = {k: torch.from_numpy(v[not_tar_y, ...]) for k, v in obs_dict.items()}
        act_dict_v = {k: v[not_tar_y, ...] for k, v in act_dict.items()}
        rew_vec_v = torch.from_numpy(rew_vec[not_tar_y, ...])
        is_fir_v = torch.ones(len(not_tar_y)) if is_first else torch.zeros(len(not_tar_y))
        done_v = (done_vec.to(torch.bool) | dead_agents_vec.to(torch.bool)).to(done_vec.dtype)[not_tar_y, ...] # 경기 종료 혹은 next-state에서 막 죽은 에이전트는 done 처리

        if not_tar_y.size > 0:
            rollout_dict = {
                "id": id_v,
                "obs_dict": obs_dict_v,
                "act_dict": act_dict_v,
                "rew_vec": rew_vec_v,
                "is_fir": is_fir_v,
                "done": done_v,
            }

        return rollout_dict

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
            # agent_tag = [] # TODO: 디버그 완료 후, 제거 필요
            # is_full = [False] # TODO: 디버그 완료 후, 제거 필요
            for _ in range(self.env_info["episode_limit"]):
                obs_dict = self.env.get_obs_dict()
                act_dict = self.model.act(obs_dict, hx, cx)
                self.set_default_actions(act_dict)
                
                # check_act_avail(self.env, act_dict, self.env_info["n_agents"])
                # check_agent_id(self.env.agents, self.env_info["n_agents"], agent_tag, is_full)
                
                rew_vec, terminated, info = self.env.step_dict(act_dict, dead_agents_vec)
                # self.env.render() # Uncomment for rendering

                self.epi_rew_vec += rew_vec
                done_vec = torch.ones(self.env_info["n_agents"]) if terminated else torch.zeros(self.env_info["n_agents"])
                
                roll_out = self.create_rollout_dict(_id, obs_dict, act_dict, rew_vec, is_first, done_vec, dead_agents_vec)
                # roll_out = self.create_rollout_dict_except_already_dead(_id, obs_dict, act_dict, rew_vec, is_first, done_vec, dead_agents_vec)
                if roll_out is not None and isinstance(roll_out, dict):
                    await self.pub_rollout(**roll_out)

                # self.env.update_death_tracker()

                is_first = False
                hx = act_dict["hx"]
                cx = act_dict["cx"]
                
                await asyncio.sleep(0.15)

                if self.heartbeat is not None:
                    self.heartbeat.value = time.monotonic()

                if terminated:
                    # if not info.get("battle_won", True):
                    #     assert bool(dead_agents_vec.all()) # 패배한 경우, 모든 에이전트는 반드시 사망해야 함
                    break

            await self.pub_stat(info) # 경기 종료 시 반환하는 info 포함
            
            
class TestWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def render_test(self):
        print(f"Build Environment for {self.worker_name}")

        while not self.stop_event.is_set():
            self.env.reset()

            hx, cx = self.initialize_lstm()

            dead_agents_vec = torch.zeros(self.env_info["n_agents"])
            
            agent_tag = [] # TODO: 디버그 완료 후, 제거 필요
            is_full = [False] # TODO: 디버그 완료 후, 제거 필요
            
            for _ in range(self.env_info["episode_limit"]):
                obs_dict = self.env.get_obs_dict()
                act_dict = self.model.act(obs_dict, hx, cx)
                self.set_default_actions(act_dict)
                
                check_act_avail(self.env, act_dict, self.env_info["n_agents"])
                check_agent_id(self.env.agents, self.env_info["n_agents"], agent_tag, is_full)
                
                rew_vec, terminated, info = self.env.step_dict(act_dict, dead_agents_vec)
                self.env.render() # Uncomment for rendering
                
                print(f"rew_vec : {rew_vec}")
                
                # self.env.update_death_tracker()
                
                hx = act_dict["hx"]
                cx = act_dict["cx"]
                
                time.sleep(0.15)

                if terminated:
                    # if not info.get("battle_won", True):
                    #     assert bool(dead_agents_vec.all()) # 패배한 경우, 모든 에이전트는 반드시 사망해야 함
                    break
