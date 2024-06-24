import torch
import time

from collections import defaultdict
from heapq import heappush, heappop
from buffers.trajectory import Trajectory2


def make_as_array(trajectory_obj):
    assert trajectory_obj.len > 0

    refrased_rollout_data = defaultdict(list)

    for rollout in trajectory_obj.data:
        for key, value in rollout.items():
            if key != "id":  # 학습 데이터만 취급
                refrased_rollout_data[key].append(value)

    refrased_rollout_data = {
        k: torch.stack(v, 0) for k, v in refrased_rollout_data.items()
    }
    return refrased_rollout_data


def rearrange_data(data):
    arranged_data = defaultdict(dict)
    
    ids = data["id"] # 각 경기 별 고유 에이전트 id
    obs_dict = data["obs_dict"]
    act_dict = data["act_dict"]
    rew_vec = data["rew_vec"]
    is_fir = data["is_fir"]
    done = data["done"]

    for idx, a_id in enumerate(ids):
        arranged_data[a_id].update({k: v[idx] for k, v in obs_dict.items()})  # (dim, )
        arranged_data[a_id].update({k: v[idx] for k, v in act_dict.items()})  # (dim, )

        arranged_data[a_id]["rew_vec"] = rew_vec[idx] # (rev_d, )
        arranged_data[a_id]["is_fir"] = is_fir[idx].unsqueeze(-1) # (1, )
        arranged_data[a_id]["done"] = done[idx].unsqueeze(-1) # (1, )
    
    return arranged_data


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

        arranged_data = rearrange_data(data)

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
                    data["is_fir"] = torch.tensor([1.0])
                else:
                    tj_ = Trajectory2(
                        self.seq_len, time.monotonic()
                    )  # Trajectory 객체 신규 생성을 통한 할당

                tj_.put(data)
                self.roll_q[a_id] = tj_

            # 롤아웃 시퀀스 길이가 충족된 경우
            if self.roll_q[a_id].len >= self.seq_len:
                await self.ready_roll.put(make_as_array(self.roll_q.pop(a_id)))
            else:
                # 롤아웃 시퀀스가 종료된 경우
                if done:
                    self.roll_q_done[a_id] = self.roll_q.pop(a_id)
