from .learner_module.learner_lib import learning as alearning
from .learner_module.ppo.ppo_learning import train_step as ppo_train_step
from .learner_module.impala.impala_learning import train_step as impala_train_step


def ppo_awrapper(timer):
    def _outer(func):  # 주의) func 자체는 껍데기 동기함수
        async def _inner(self, *args, **kwargs):
            return await alearning(self, ppo_train_step, timer, *args, **kwargs)

        return _inner

    return _outer


def impala_awrapper(timer):
    def _outer(func):  # 주의) func 자체는 껍데기 동기함수
        async def _inner(self, *args, **kwargs):
            return await alearning(self, impala_train_step, timer, *args, **kwargs)

        return _inner

    return _outer