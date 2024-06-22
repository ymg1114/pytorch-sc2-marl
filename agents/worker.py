import uuid
import time
import zmq
import asyncio
import torch
# import torch.jit as jit
import numpy as np

from utils.utils import Protocol, SC2Config, encode, decode

from env.sc2_env_wrapper import WrapperSMAC2

from rewarder.rewarder import REWARD_PARAM


class Worker:
    def __init__(
        self, args, model, worker_name, stop_event, manager_ip, learner_ip, port, learner_port, heartbeat=None
    ):
        self.args = args
        self.device = args.device  # cpu

        self.model = model.to(torch.device("cpu")).eval()
        self.worker_name = worker_name
        self.stop_event = stop_event
        self.heartbeat = heartbeat

        self.env = WrapperSMAC2(
            capability_config=SC2Config,
            map_name="10gen_terran",
            debug=True,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
        )

        self.env_info = self.env.get_env_info()
        self.env_space = self.env.get_env_space()

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
                if model_state_dict:
                    self.model.load_state_dict(
                        model_state_dict
                    )  # reload learned-model from learner

            await asyncio.sleep(0.1)

    async def pub_rollout(self, **roll_out):
        await self.pub_socket.send_multipart([*encode(Protocol.Rollout, roll_out)])

    async def pub_stat(self):
        await self.pub_socket.send_multipart([*encode(Protocol.Stat, self.epi_rew_vec)])
        print(
            f"worker_name: {self.worker_name} epi_rew_vec: {self.epi_rew_vec} pub stat to manager!"
        )

    async def life_cycle_chain(self):
        tasks = [
            asyncio.create_task(self.collect_rolloutdata()),
            asyncio.create_task(self.req_model()),
        ]
        await asyncio.gather(*tasks)

    async def collect_rolloutdata(self):
        print("Build Environment for {}".format(self.worker_name))

        # self.num_epi = 0
        
        while not self.stop_event.is_set():
            self.env.reset()

            _id = str(uuid.uuid4())  # 고유한 난수 생성

            lstm_hx = (
                torch.zeros(self.env_info["n_agents"], self.args.hidden_size),
                torch.zeros(self.env_info["n_agents"], self.args.hidden_size),
            )
            
            self.epi_rew_vec = np.zeros((self.env_info["n_agents"], len(REWARD_PARAM)), dtype=np.float32)
            
            dead_agents_vec = torch.zeros(self.env_info["n_agents"])
            is_fir = True  # first frame
            
            for _ in range(self.env_info["episode_limit"]):
                # act, logits, log_prob, lstm_hx_next = self.model.act(obs_dict, lstm_hx)
                # next_obs, rew, done = self.env.step(act.item())
                
                obs_dict = self.env.get_obs_dict()
                act_dict = self.model.act(obs_dict, lstm_hx)
                
                assert "on_select_act" in self.env_space["act"]
                assert "on_select_move" in self.env_space["act"]
                assert "on_select_target" in self.env_space["act"]
                
                act_dict["on_select_act"] = torch.ones(self.env_info["n_agents"]) # 항상 선택
                act_dict["on_select_move"] = torch.zeros(self.env_info["n_agents"])
                act_dict["on_select_target"] = torch.zeros(self.env_info["n_agents"])
                
                rew_vec, terminated, info = self.env.step(act_dict, dead_agents_vec)
                # env.render() # Uncomment for rendering
                self.epi_rew_vec += rew_vec
                
                done_vec = torch.ones(self.env_info["n_agents"]) if terminated else torch.zeros(self.env_info["n_agents"])
                
                roll_out = {
                    "id": [f"{_id}_{agent_id}" for agent_id in range(self.env_info["n_agents"])],
                    "obs_dict": obs_dict,
                    "act_dict": act_dict,
                    "rew_vec": rew_vec,
                    "is_fir": torch.ones(self.env_info["n_agents"]) if is_fir else torch.zeros(self.env_info["n_agents"]),
                    "done": done_vec | dead_agents_vec, # 경기 종료 혹은 죽은 에이전트는 done 처리
                }
                
                # roll_out = {
                #     "obs": obs,
                #     "act": act.view(-1),
                #     "rew": torch.from_numpy(
                #         np.array([rew * self.args.reward_scale])
                #     ),
                #     "logits": logits,
                #     "log_prob": log_prob.view(-1),
                #     "is_fir": torch.FloatTensor([1.0 if is_fir else 0.0]),
                #     "done": torch.FloatTensor([1.0 if done else 0.0]),
                #     "hx": lstm_hx[0],
                #     "cx": lstm_hx[1],
                #     "id": _id,
                # }

                await self.pub_rollout(**roll_out)

                is_fir = False
                lstm_hx = act_dict["lstm_hxs"]
                
                # obs = next_obs
                # lstm_hx = lstm_hx_next

                await asyncio.sleep(0.15)

                if self.heartbeat is not None:
                    self.heartbeat.value = time.time()

                if terminated:
                    if info["battle_won"] is False:
                        assert dead_agents_vec.all() # 패배한 경우, 모든 에이전트는 반드시 사망해야 함
                    break

            await self.pub_stat()
            # self.num_epi += 1
