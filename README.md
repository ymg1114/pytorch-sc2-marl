# pytorch-sc2-marl
## Architecture of Multi-Agents Distributed Reinforcement Learning
![Distributed Reinforcement Learning](https://github.com/user-attachments/assets/f471b1bc-f4da-43b9-893d-cdd8f3811560)  
![Worker Process](https://github.com/user-attachments/assets/63b3ad0e-5e14-4bcb-ba66-b1e99c4803af)  

## Algo
__PPO__  
__IMPALA__  
In the [parameters.json](https://github.com/ymg1114/pytorch-sc2-marl/blob/main/utils/parameters.json) configuration file, you can replace the algorithm by changing the `algo` entry.  

## Caution!
This repository targets the [SMAC2](https://github.com/oxwhirl/smacv2) environment.  
And, this repository has been validated for training on `5vs5 sc2 marine battles`.  
You must ensure the complete installation of the __SMAC2__ environment on the machine that will be used as a Worker.  

## How to run
This repository is developed assuming the use of __tmux-interactive-session__ and a __Python Miniconda__ environment on an __Ubuntu machine__.  
In the [run.py](https://github.com/ymg1114/pytorch-sc2-marl/blob/main/run.py) script, you need to set the virtual environment name `vir_env_name`.  
In the [machines.json](https://github.com/ymg1114/pytorch-sc2-marl/blob/main/utils/machines.json) configuration file, you need to set the `account`, `IP`, and `port` for the machines to be used as __Learner__, __Manager__, and __Worker__.   
Additionally, in the Worker machine settings, you need to decide `how many independent worker processes` to use.  
After all settings are properly configured, you need to run `python run.py`
