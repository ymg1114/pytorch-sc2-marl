import ctypes
import numpy as np

from utils.utils import *

from typing import TYPE_CHECKING
from utils.utils import Position, lib

if TYPE_CHECKING:
    from observer.observer import Observer


class Avail():
    def __init__(self, observer: "Observer"):
        self.observer = observer

    def pre_avail_actions(self):
        """가공전 avail 정보"""
        
        # TODO: FLEE 행동은 커스텀 디자인 행동이기 때문에, avail_total_act -> 에서는 avail 여부를 확인할 수 없음
        avail_total_act = np.array(self.observer.env_core.get_avail_actions(), dtype=np.float32)
        assert avail_total_act.shape == (self.observer.n_agents, self.observer.n_actions)
        return avail_total_act

    def update_info_for_flee(self):
        self.n_ally = self.observer.n_agents
        self.n_enemy = self.observer.n_enemies
        self.n_rows = self.observer.env_core.map_y
        self.n_cols = self.observer.env_core.map_x

        # Flee 알고리즘의 하이퍼 파라미터
        self.min_search_length = 2
        self.max_search_length = 4
        self.ally_weight = 1.0
        self.enemy_weight = -2.5

        # Output array for flee positions
        self.flee_positions_np = np.zeros(self.n_ally, dtype=[("y", np.int32), ("x", np.int32)])
        
        self.positions_allies_np = np.zeros(self.n_ally, dtype=[("y", np.int32), ("x", np.int32)])
        self.positions_enemies_np = np.zeros(self.n_enemy, dtype=[("y", np.int32), ("x", np.int32)])

        allies_hp_np = np.zeros(self.n_ally, dtype=[("hp", np.int32)])
        enemies_hp_np = np.zeros(self.n_enemy, dtype=[("hp", np.int32)])

        for i, al_id in enumerate(range(self.n_ally)):
            a_unit = self.observer.env_core.get_unit_by_id(al_id)
            
            self.positions_allies_np[i] = (a_unit.pos.y, a_unit.pos.x)
            allies_hp_np[i] = a_unit.health

        for j, (e_id, e_unit) in enumerate(self.observer.env_core.enemies.items()):
            self.positions_enemies_np[j] = (e_unit.pos.y, e_unit.pos.x)
            enemies_hp_np[j] = e_unit.health

        self.alive_allies_np = allies_hp_np["hp"] > 0
        self.alive_enemies_np = enemies_hp_np["hp"] > 0

        # 갈 수 있는 지역: 0, 갈 수 없는 지역: 1
        assert self.observer.env_core.pathing_grid.shape == (self.n_rows, self.n_cols)

        grid_maps_np = np.tile(
            self.observer.env_core.pathing_grid, (self.n_ally, 1, 1)
        ).astype(np.int32)
        grid_maps_np = 1 - grid_maps_np  # 0 -> 갈 수 있음, 1 -> 갈 수 없음

        for ally in self.positions_allies_np:
            grid_maps_np[:, ally["y"], ally["x"]] = 1  # Mark allies' positions as obstacles

        for enemy in self.positions_enemies_np:
            grid_maps_np[:, enemy["y"], enemy["x"]] = 1  # Mark enemies' positions as obstacles

        self.grid_maps_ctypes = (ctypes.POINTER(ctypes.c_int) * self.n_ally)()
        for i in range(self.n_ally):
            self.grid_maps_ctypes[i] = grid_maps_np[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        # Call the function
        # 프로세스 레벨의 전역 객체로 사용
        lib.compute_flee_positions(
            ctypes.c_int(self.n_ally),
            ctypes.c_int(self.n_enemy),
            ctypes.c_int(self.n_rows),
            ctypes.c_int(self.n_cols),
            self.grid_maps_ctypes,
            self.alive_allies_np.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            self.alive_enemies_np.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            self.positions_allies_np.ctypes.data_as(ctypes.POINTER(Position)),
            self.positions_enemies_np.ctypes.data_as(ctypes.POINTER(Position)),
            ctypes.c_int(self.min_search_length),
            ctypes.c_int(self.max_search_length),
            ctypes.c_float(self.ally_weight),
            ctypes.c_float(self.enemy_weight),
            self.flee_positions_np.ctypes.data_as(ctypes.POINTER(Position))  # Output array
        )


    def avail_act_parsing(self, avail_total_act):
        """주의) 하드 코드 성향이 존재
        no-op, stop, move-north, move-south, move-east, move-west, target~ 
        순으로 avail_total_act (2d-ndarray) 가 구성됨. 이를 각 타겟 네트워크에 알맞게 슬라이싱
        
        여기서 "target~": 아군 혹은 적군의, 초기 생성 최대 인원수 까지를 -> 타겟팅 가능 인덱스로 지정
        """
        
        avail_act = np.zeros((self.observer.n_agents, self.observer.dim_act), dtype=np.float32) # (n_ally, dim_act)
        avail_act[:, NO_OP_IDX] = avail_total_act[:, NO_OP_IDX] # no-op
        avail_act[:, STOP_IDX] = avail_total_act[:, STOP_IDX] # stop
        avail_act[:, MOVE_IDX] = avail_total_act[:, MOVE_IDX: MOVE_IDX+self.observer.dim_move].any(-1) # move
        avail_act[:, TARGET_IDX] = avail_total_act[:, MOVE_IDX+self.observer.dim_move:-1].any(-1) # target / TODO: FLEE 행동 대처

        # TODO: -9999는 하드코드 임
        valid_flee_mask = (self.flee_positions_np["y"] != -9999) & (self.flee_positions_np["x"] != -9999)
        avail_act[valid_flee_mask, FLEE_IDX] = 1.0
        return avail_act

    def avail_move_parsing(self, avail_total_act):
        """주의) 하드 코드 성향이 존재
        no-op, stop, move-north, move-south, move-east, move-west, target~ 
        순으로 avail_total_act (2d-ndarray) 가 구성됨. 이를 각 타겟 네트워크에 알맞게 슬라이싱
        
        여기서 "target~": 아군 혹은 적군의, 초기 생성 최대 인원수 까지를 -> 타겟팅 가능 인덱스로 지정
        """
        
        avail_move = np.zeros((self.observer.n_agents, self.observer.dim_move), dtype=np.float32) # (n_ally, dim_move)
        avail_move[:, 0] = avail_total_act[:, MOVE_NORTH_IDX] # north
        avail_move[:, 1] = avail_total_act[:, MOVE_SOUTH_IDX] # south
        avail_move[:, 2] = avail_total_act[:, MOVE_EAST_IDX] # east
        avail_move[:, 3] = avail_total_act[:, MOVE_WEST_IDX] # west

        return avail_move

    def avail_target_parsing(self, avail_total_act):
        """주의) 하드 코드 성향이 존재
        no-op, stop, move-north, move-south, move-east, move-west, target~ 
        순으로 avail_total_act (2d-ndarray) 가 구성됨. 이를 각 타겟 네트워크에 알맞게 슬라이싱
        
        여기서 "target~": 아군 혹은 적군의, 초기 생성 최대 인원수 까지를 -> 타겟팅 가능 인덱스로 지정
        """
        
        avail_target = np.zeros((self.observer.n_agents, self.observer.dim_target), dtype=np.float32) # (n_ally, dim_target)
        avail_target[:, :] = avail_total_act[:, MOVE_IDX+self.observer.dim_move:-1] # targets / TODO: FLEE 행동 대처

        return avail_target

    def get_avail(self):
        avail_total_act = self.pre_avail_actions()
        self.update_info_for_flee()
        
        avail_dict = {
            "avail_act": self.avail_act_parsing(avail_total_act),
            "avail_move": self.avail_move_parsing(avail_total_act),
            "avail_target": self.avail_target_parsing(avail_total_act),
        }
        return avail_dict
