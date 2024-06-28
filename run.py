import os

from utils.utils import Machines

"""SSH 인증서 및 파이썬 아나콘다 가상 환경에 대한 설정이 되어있다고 가정
"""


vir_env_name = "..." # 모든 분산 머신에서 공용

# 접속 설정
local_dir = os.getcwd()
remote_dir = "~/remote_repo_sc2"
ssh_private_key = "ssh -i ~/.ssh/id_rsa" # ssh 개인키, 접속 대상 머신에 공개키를 머리 전송해 놓아야 함

exclude_dirs = ["results", "logs", "assets", "__pycache__", "LICENSE", "README.md", ".git", ".gitignore"]
pre_activate = "source ~/anaconda3/etc/profile.d/conda.sh" # 모든 분산 머신에서 공용
post_activet_env = f"conda activate {vir_env_name}" # 모든 분산 머신에서 공용


# 복사 제외 디렉터리 옵션 생성
exclude_opts = ' '.join([f"--exclude={d}" for d in exclude_dirs])


def append_command(commands, new_command):
    return commands + new_command + "\n"


def start_tmux_session(commands, session_name):
    return append_command(commands, f"tmux new-session -d -s {session_name}")


def ssh_connect(commands, session_name, account, remote_ip):
    return append_command(commands, f'tmux send-keys -t {session_name} "ssh {account}@{remote_ip}" C-m')


def make_remote_directory(commands, session_name, remote_dir):
    return append_command(commands, f'tmux send-keys -t {session_name} "mkdir -p {remote_dir}" C-m')


def copy_directory(commands, session_name, exclude_opts, ssh_private_key, local_dir, account, remote_ip, remote_dir):
    return append_command(commands, f'tmux send-keys -t {session_name} "rsync -avz --progress {exclude_opts} -e {ssh_private_key} {local_dir}/ {account}@{remote_ip}:{remote_dir}/" C-m')


def activate_vir_env(commands, session_name, pre_activate, post_activet_env):
    return append_command(commands, f'tmux send-keys -t {session_name} "{pre_activate} && {post_activet_env}" C-m')


def run_python_script(commands, session_name, remote_dir, run_python):
    return append_command(commands, f'tmux send-keys -t {session_name} "cd {remote_dir} && {run_python}" C-m')


def exit_tmux_session(commands, session_name, should_exit=True):
    if should_exit:
        commands = append_command(commands, f'tmux send-keys -t {session_name} "exit" C-m') # SSH 세션을 종료
    return append_command(commands, f"tmux detach -s {session_name}") # tmux 세션을 분리 -> 현재 터미널로 회귀


if __name__ == "__main__":
    commands = ""

    learner_info = Machines.learner
    worker_infos = Machines.workers
    
    # Learner
    session_name = f"learner_{learner_info.ip.replace('.', '_')}"
    # run_python = f"nohup python main.py learner_sub_process {learner_info.ip} {learner_info.port} learner.log 2>&1 &"
    run_python = f"python main.py learner_sub_process {learner_info.ip} {learner_info.port}"
    
    commands = start_tmux_session(commands, session_name)
    commands = copy_directory(commands, session_name, exclude_opts, ssh_private_key, local_dir, learner_info.account, learner_info.ip, remote_dir)
    commands = ssh_connect(commands, session_name, learner_info.account, learner_info.ip)
    commands = make_remote_directory(commands, session_name, remote_dir)
    commands = activate_vir_env(commands, session_name, pre_activate, post_activet_env)
    commands = run_python_script(commands, session_name, remote_dir, run_python)
    # commands = exit_tmux_session(commands, session_name)
    
    for i, worker_info in enumerate(worker_infos):
        # Manager
        session_name = f"manager_{i}_{worker_info.manager_ip.replace('.', '_')}"
        # run_python = f"nohup python main.py manager_sub_process {worker_info.manager_ip} {learner_info.ip} {worker_info.port} {learner_info.port} manager_{i}.log 2>&1 &"
        run_python = f"python main.py manager_sub_process {worker_info.manager_ip} {learner_info.ip} {worker_info.port} {learner_info.port}"
        
        commands = start_tmux_session(commands, session_name)
        commands = copy_directory(commands, session_name, exclude_opts, ssh_private_key, local_dir, worker_info.account, worker_info.manager_ip, remote_dir)
        commands = ssh_connect(commands, session_name, worker_info.account, worker_info.manager_ip)
        commands = make_remote_directory(commands, session_name, remote_dir)
        commands = activate_vir_env(commands, session_name, pre_activate, post_activet_env)
        commands = run_python_script(commands, session_name, remote_dir, run_python)
        # commands = exit_tmux_session(commands, session_name)

        # Worker
        session_name = f"worker_{i}_{worker_info.ip.replace('.', '_')}"
        # run_python = f"nohup python main.py worker_sub_process {worker_info.num_p} {worker_info.manager_ip} {learner_info.ip} {worker_info.port} {learner_info.port} worker_{i}.log 2>&1 &"
        run_python = f"python main.py worker_sub_process {worker_info.num_p} {worker_info.manager_ip} {learner_info.ip} {worker_info.port} {learner_info.port}"
        
        commands = start_tmux_session(commands, session_name)
        commands = copy_directory(commands, session_name, exclude_opts, ssh_private_key, local_dir, worker_info.account, worker_info.ip, remote_dir)
        commands = ssh_connect(commands, session_name, worker_info.account, worker_info.ip)
        commands = make_remote_directory(commands, session_name, remote_dir)
        commands = activate_vir_env(commands, session_name, pre_activate, post_activet_env)
        commands = run_python_script(commands, session_name, remote_dir, run_python)
        # commands = exit_tmux_session(commands, session_name)

    # 스크립트 실행
    os.system(commands)
    print(f"commands: \n{commands}")