import zmq
import zmq.asyncio

import asyncio
import numpy as np
from collections import deque

from utils.utils import Protocol, encode, decode, extract_values, extract_nested_values


class Manager:
    def __init__(self, args, stop_event, manager_ip, learner_ip, port, learner_port):
        self.args = args
        self.stop_event = stop_event

        self.data_q = deque(maxlen=1024)

        self.stat_publish_cycle = 50
        self.stat_q = deque(maxlen=self.stat_publish_cycle)

        self.zeromq_set(manager_ip, learner_ip, port, learner_port)

    def __del__(self):  # 소멸자
        if hasattr(self, "pub_socket"):
            self.pub_socket.close()
        if hasattr(self, "sub_socket"):
            self.sub_socket.close()

    def zeromq_set(self, manager_ip, learner_ip, port, learner_port):
        context = zmq.asyncio.Context()

        # worker <-> manager
        self.sub_socket = context.socket(zmq.SUB)  # subscribe rollout-data, stat-data
        self.sub_socket.bind(f"tcp://{manager_ip}:{port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

        # manager <-> learner-storage
        self.pub_socket = context.socket(zmq.PUB)  # publish batch-data, stat-data
        self.pub_socket.connect(f"tcp://{learner_ip}:{learner_port}")

    async def sub_data(self):
        while not self.stop_event.is_set():
            protocol, data = decode(*await self.sub_socket.recv_multipart())
            if len(self.data_q) == self.data_q.maxlen:
                self.data_q.popleft()  # FIFO
            self.data_q.append((protocol, data))

            await asyncio.sleep(0.001)

    async def pub_data(self):
        stat_pub_num = 0  # 지역 변수

        while not self.stop_event.is_set():
            if len(self.data_q) > 0:
                protocol, data = self.data_q.popleft()  # FIFO
                if protocol is Protocol.Rollout:
                    await self.pub_socket.send_multipart(
                        [*encode(Protocol.Rollout, data)]
                    )

                elif protocol is Protocol.Stat:
                    self.stat_q.append(data)
                    if stat_pub_num >= self.stat_publish_cycle and len(self.stat_q) > 0:
                        _epi_rew_vec = extract_values(self.stat_q, "epi_rew_vec")
                        
                        _battle_won = extract_nested_values(self.stat_q, "info", "battle_won")
                        _dead_allies = extract_nested_values(self.stat_q, "info", "dead_allies")
                        _dead_enemies = extract_nested_values(self.stat_q, "info", "dead_enemies")
  
                        await self.pub_socket.send_multipart(
                            [
                                *encode(
                                    Protocol.Stat,
                                    {
                                        "log_len": len(self.stat_q),
                                        "mean_battle_won":np.mean(_battle_won),
                                        "mean_dead_allies":np.mean(_dead_allies),
                                        "mean_dead_enemies":np.mean(_dead_enemies),
                                        "mean_rew_vec": np.mean(_epi_rew_vec, axis=(0, 1)), # mean epi_rew_vec in REWARD_PARAM-wise
                                    },
                                )
                            ]
                        )
                        stat_pub_num = 0
                    stat_pub_num += 1
                else:
                    assert False, f"Wrong protocol: {protocol}"

            await asyncio.sleep(0.001)

    async def data_chain(self):
        tasks = [
            asyncio.create_task(self.sub_data()),
            asyncio.create_task(self.pub_data()),
        ]
        await asyncio.gather(*tasks)
