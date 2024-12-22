import time

import jax.numpy as jnp

from heapq import heappush, heappop
from buffers.trajectory import Trajectory2

from .rollout_assembler_lib import make_as_array_jax, rearrange_data_origin


class RolloutAssembler:
    def __init__(self, args, ready_roll):
        self.args = args
        self.seq_len = args.seq_len
        self.roll_q = dict()
        self.roll_q_done = dict()
        self.ready_roll = ready_roll

    async def pop(self):
        return await self.ready_roll.get()

    async def push(self, data):
        assert "id" in data
        assert "obs_dict" in data
        assert "act_dict" in data
        assert "rew_vec" in data
        assert "is_fir" in data
        assert "done" in data

        arranged_data = rearrange_data_origin(data)

        # Trajectory 객체 최초 생성 이후 2.0초가 지나면 삭제. policy-lag를 줄이기 위함.
        self.roll_q = {
            id: tj
            for id, tj in self.roll_q.items()
            if time.monotonic() - tj.gen_time < 2.0
        }

        for a_id in arranged_data:
            data = arranged_data[a_id]
            done = data["done"]
            
            if a_id in self.roll_q:  # 기존 경기에 데이터 추가
                self.roll_q[a_id].put(data)
            else:
                if len(self.roll_q_done) > 0:
                    _, aid_ = heappop(
                        [(tj.len, aid) for aid, tj in self.roll_q_done.items()]
                    )  # 데이터의 크기 (roll 개수)가 가장 작은 Trajectory 추출
                    tj_ = self.roll_q_done.pop(aid_)
                    data["is_fir"] = jnp.array([1.0])
                else:
                    tj_ = Trajectory2(
                        self.seq_len, time.monotonic()
                    )  # Trajectory 객체 신규 생성을 통한 할당

                tj_.put(data)
                self.roll_q[a_id] = tj_

            # 롤아웃 시퀀스 길이가 충족된 경우
            if self.roll_q[a_id].len >= self.seq_len:
                await self.ready_roll.put(make_as_array_jax(self.roll_q.pop(a_id)))
            else:
                # 롤아웃 시퀀스가 종료된 경우
                if done:
                    self.roll_q_done[a_id] = self.roll_q.pop(a_id)
