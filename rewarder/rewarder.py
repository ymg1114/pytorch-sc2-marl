import numpy as np

from utils.utils import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smacv2.env.starcraft2.starcraft2 import StarCraft2Env


# TODO: 리워드 파라미터 (가중치) 변경 필요
REWARD_PARAM = {
    "my_health": 0.007,
    "my_death": -0.7,
    "ally_health": 0.002,
    "ally_death": -0.2,
    "enemy_health": -0.004,
    "enemy_death": 0.4,
    "win": 2.0,
    "lose": 0.0,
}

r_func = {}


def register(name):
    assert name in REWARD_PARAM

    def _wrapper(func):
        r_func[name] = func
        return func
    return _wrapper


@register(name="my_health")
def cal_my_health(env_core, rdx, rew_vec, **kwargs):
    """내 현재 체력 (+쉴드) 변화량 확인"""
    
    for al_id, al_unit in env_core.agents.items():
        if not env_core.death_tracker_ally[al_id]: # did not die so far
            prev_health = (
                env_core.previous_ally_units[al_id].health
                + env_core.previous_ally_units[al_id].shield
            )
            if al_unit.health == 0:
                # just died
                rew_vec[rdx, al_id] = -prev_health
            else:
                # still alive
                rew_vec[rdx, al_id] = al_unit.health + al_unit.shield - prev_health


@register(name="my_death")
def cal_my_death(env_core, rdx, rew_vec, **kwargs):
    """내가 막 죽었는지 확인"""
    
    for al_id, al_unit in env_core.agents.items():
        if not env_core.death_tracker_ally[al_id]: # did not die so far
            if al_unit.health == 0:
                # just died
                rew_vec[rdx, al_id] = 1.0


@register(name="ally_health")
def cal_ally_health(env_core, rdx, rew_vec, **kwargs):
    """현재 아군 체력 (+쉴드) 총 변화량 확인"""
    
    ally_health = 0.0
    for al_id, al_unit in env_core.agents.items():
        if not env_core.death_tracker_ally[al_id]: # did not die so far
            prev_health = (
                env_core.previous_ally_units[al_id].health
                + env_core.previous_ally_units[al_id].shield
            )
            if al_unit.health == 0:
                # just died
                ally_health += -prev_health
            else:
                # still alive
                ally_health += al_unit.health + al_unit.shield - prev_health

    rew_vec[rdx, :] = ally_health


@register(name="ally_death")
def cal_ally_death(env_core, rdx, rew_vec, **kwargs):
    """막 죽은 아군 개체수 확인"""
    
    death_ally = 0.0
    for al_id, al_unit in env_core.agents.items():
        if not env_core.death_tracker_ally[al_id]: # did not die so far
            if al_unit.health == 0:
                # just died
                death_ally += 1.0

    rew_vec[rdx, :] = death_ally
    
    
@register(name="enemy_health")
def cal_enemy_health(env_core, rdx, rew_vec, **kwargs):
    """현재 적군 체력 (+쉴드) 총 변화량 확인"""
    
    enemy_health = 0.0
    for e_id, e_unit in env_core.enemies.items():
        if not env_core.death_tracker_enemy[e_id]: # did not die so far
            prev_health = (
                env_core.previous_enemy_units[e_id].health
                + env_core.previous_enemy_units[e_id].shield
            )
            if e_unit.health == 0:
                # just died
                enemy_health += -prev_health
            else:
                enemy_health += e_unit.health + e_unit.shield - prev_health

    rew_vec[rdx, :] = enemy_health


@register(name="enemy_death")
def cal_enemy_death(env_core, rdx, rew_vec, **kwargs):
    """막 죽은 적군 개체수 확인"""
    
    death_enemy = 0.0
    for e_id, e_unit in env_core.enemies.items():
        if not env_core.death_tracker_enemy[e_id]: # did not die so far
            if e_unit.health == 0:
                # just died
                death_enemy += 1.0

    rew_vec[rdx, :] = death_enemy


@register(name="win")
def cal_win(env_core, rdx, rew_vec, **kwargs):
    game_end_code = kwargs.get("game_end_code", "")
    assert game_end_code != ""
    
    if game_end_code is not None and game_end_code == 1:
        rew_vec[rdx, :] = 1.0 # 승리


@register(name="lose")
def cal_lose(env_core, rdx, rew_vec, **kwargs):
    game_end_code = kwargs.get("game_end_code", "")
    assert game_end_code != ""
    
    if game_end_code is not None and game_end_code == -1:
        rew_vec[rdx, :] = 1.0 # 패배


class Rewarder():
    def __init__(self, env_core: "StarCraft2Env"):
        self.env_core = env_core # 상위 SC2Env

    def update(self, rew_vec):
        """리워드 연산 후, 자료형 업데이트"""
        
        for al_id, al_unit in self.env_core.agents.items():
            if self.env_core.death_tracker_ally[al_id]: # already dead
                rew_vec[:, al_id] *= 0.0 # 이미 죽어있는 아군 개체에 대해서는 더 이상 리워드를 연산하지 않음
            else: # did not die so far
                if al_unit.health == 0:
                    # just died
                    self.env_core.death_tracker_ally[al_id] = 1 # 자료형 업데이트

        for e_id, e_unit in self.env_core.enemies.items():
            if not self.env_core.death_tracker_enemy[e_id]: # did not die so far
                if e_unit.health == 0:
                    # just died
                    self.env_core.death_tracker_enemy[e_id] = 1 # 자료형 업데이트

    def get(self, game_end_code):
        """주의) 기존에 죽은 아군 유닛에 대해서는, 리워드를 더 이상 연산하지 않음
        TODO: 중복 로직 최적화 필요
        """
        
        rew_vec = np.zeros((len(REWARD_PARAM), len(self.env_core.agents)), dtype=np.float32) # 리워드 벡터 초기화, (REWARD, Agents)

        for rdx, name in enumerate(REWARD_PARAM):
            assert name in r_func
            r_func[name](self.env_core, rdx, rew_vec, game_end_code=game_end_code)

        self.update(rew_vec)

        return rew_vec.T # (Agents, REWARD)
