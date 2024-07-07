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
activet_env = f"conda activate {vir_env_name}" # 모든 분산 머신에서 공용


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


def activate_vir_env(commands, session_name, activet_env):
    return append_command(commands, f'tmux send-keys -t {session_name} "{activet_env}" C-m')


def run_python_script(commands, session_name, remote_dir, run_python):
    return append_command(commands, f'tmux send-keys -t {session_name} "cd {remote_dir} && {run_python}" C-m')


def exit_tmux_session(commands, session_name, should_exit=True):
    if should_exit:
        commands = append_command(commands, f'tmux send-keys -t {session_name} "exit" C-m') # SSH 세션을 종료
    return append_command(commands, f"tmux detach -s {session_name}") # tmux 세션을 분리 -> 현재 터미널로 회귀


if __name__ == "__main__":
    commands = ""

    infos = Machines.storage
    learner_account = Machines.learner_account
    learner_ip = Machines.learner_ip
    learner_worker_port = Machines.learner_worker_port  # Learner와 Worker 사이의 stat 및 학습된 신경망 전송을 위한 port

    # Learner와 Storage는 반드시 동일 머신에 존재
    # Learner는 1개, Storage는 여러 개가 될 수 있음 -> Learner ip는 1개, Storage 객체의 port는 여러 개
    
    # Learner
    session_name = f"learner_{learner_ip.replace('.', '_')}"
    # run_python = "nohup python main.py learner_sub_process {learner_ip} {learner_worker_port} learner.log 2>&1 &"
    run_python = f"python main.py learner_sub_process {learner_ip} {learner_worker_port}"
    for i, info in enumerate(infos):
        run_python += f" {info.learner_port}"

    commands = start_tmux_session(commands, session_name)
    commands = copy_directory(commands, session_name, exclude_opts, ssh_private_key, local_dir, learner_account, learner_ip, remote_dir)
    commands = ssh_connect(commands, session_name, learner_account, learner_ip)
    commands = make_remote_directory(commands, session_name, remote_dir)
    commands = activate_vir_env(commands, session_name, activet_env)
    commands = run_python_script(commands, session_name, remote_dir, run_python)
    # commands = exit_tmux_session(commands, session_name)

    # Manager와 Worker는 반드시 동일 머신에 존재
    # Worker의 ip, port 정보는 필요치 않음 -> Manager 단에서 bind된 ip, port에 connect 하기 때문
    for i, info in enumerate(infos):
        for j, w_info in enumerate(info.workers):
            # Manager
            session_name = f"manager_{i}_{j}_{w_info.manager_ip.replace('.', '_')}"
            # run_python = f"nohup python main.py manager_sub_process {w_info.manager_ip} {learner_ip} {w_info.manager_port} {info.learner_port} > manager.log 2>&1 &"
            run_python = f"python main.py manager_sub_process {w_info.manager_ip} {learner_ip} {w_info.manager_port} {info.learner_port}"

            commands = start_tmux_session(commands, session_name)
            commands = copy_directory(commands, session_name, exclude_opts, ssh_private_key, local_dir, w_info.account, w_info.manager_ip, remote_dir)
            commands = ssh_connect(commands, session_name, w_info.account, w_info.manager_ip)
            commands = make_remote_directory(commands, session_name, remote_dir)
            commands = activate_vir_env(commands, session_name, activet_env)
            commands = run_python_script(commands, session_name, remote_dir, run_python)
            # commands = exit_tmux_session(commands, session_name)

            # Worker
            session_name = f"worker_{i}_{j}_{w_info.manager_ip.replace('.', '_')}"
            # run_python = f"nohup python main.py worker_sub_process {w_info.num_p} {w_info.manager_ip} {learner_ip} {w_info.manager_port} {learner_worker_port} > worker.log 2>&1 &"
            run_python = f"python main.py worker_sub_process {w_info.num_p} {w_info.manager_ip} {learner_ip} {w_info.manager_port} {learner_worker_port}"

            commands = start_tmux_session(commands, session_name)
            commands = copy_directory(commands, session_name, exclude_opts, ssh_private_key, local_dir, w_info.account, w_info.manager_ip, remote_dir)
            commands = ssh_connect(commands, session_name, w_info.account, w_info.manager_ip)
            commands = make_remote_directory(commands, session_name, remote_dir)
            commands = activate_vir_env(commands, session_name, activet_env)
            commands = run_python_script(commands, session_name, remote_dir, run_python)
            # commands = exit_tmux_session(commands, session_name)

    # 스크립트 실행
    os.system(commands)
    print(f"commands: \n{commands}")