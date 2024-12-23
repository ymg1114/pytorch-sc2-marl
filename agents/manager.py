import zmq
import zmq.asyncio
import time
import asyncio
import numpy as np
# from collections import deque
from utils.utils import Protocol, encode, decode


class Manager:
    def __init__(self, args, stop_event, manager_ip, learner_ip, manager_port, learner_port, heartbeat=None):
        self.args = args
        self.stop_event = stop_event

        self.heartbeat = heartbeat
        
        self.zeromq_set(manager_ip, learner_ip, manager_port, learner_port)

    def __del__(self):  # 소멸자
        if hasattr(self, "pub_socket"):
            self.pub_socket.close()
        if hasattr(self, "sub_socket"):
            self.sub_socket.close()

    def zeromq_set(self, manager_ip, learner_ip, manager_port, learner_port):
        context = zmq.asyncio.Context()

        # worker <-> manager
        self.sub_socket = context.socket(zmq.SUB)  # subscribe rollout-data, stat-data
        self.sub_socket.bind(f"tcp://{manager_ip}:{manager_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

        # manager <-> learner-storage
        self.pub_socket = context.socket(zmq.PUB)  # publish batch-data, stat-data
        self.pub_socket.connect(f"tcp://{learner_ip}:{learner_port}")

    async def sub_data(self):
        while not self.stop_event.is_set():
            protocol, data = decode(*await self.sub_socket.recv_multipart())
            await self.data_q.put((protocol, data))
            
            await asyncio.sleep(1e-4)

    async def pub_data(self):
        while not self.stop_event.is_set():
            protocol, data = await self.data_q.get()  # FIFO
            assert protocol is Protocol.Rollout

            if self.heartbeat is not None:
                self.heartbeat.value = time.monotonic()

            await self.pub_socket.send_multipart(
                [*encode(Protocol.Rollout, data)]
            )
            await asyncio.sleep(1e-4)

    async def data_chain(self):
        self.data_q = asyncio.Queue(1024)
        tasks = [
            asyncio.create_task(self.sub_data()),
            asyncio.create_task(self.pub_data()),
        ]
        await asyncio.gather(*tasks)
