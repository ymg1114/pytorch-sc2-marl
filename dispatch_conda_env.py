import os

from utils.utils import Machines

"""SSH 인증서에 대한 설정이 되어있다고 가정
miniconda3 환경을 대상으로 함.
"""


def append_command(commands, new_command):
    return commands + new_command + '\n'


def start_tmux_session(commands, session_name):
    return append_command(commands, f"tmux new-session -d -s {session_name}")


def conda_pack(commands, session_name, conda_env, local_env_file):
    return append_command(commands, f"tmux send-keys -t {session_name} 'conda pack -n {conda_env} -o {local_env_file}' C-m")


def ssh_connect(commands, session_name, account, remote_ip):
    return append_command(commands, f"tmux send-keys -t {session_name} 'ssh {account}@{remote_ip}' C-m")


def make_remote_directory(commands, session_name, remote_dir):
    return append_command(commands, f"tmux send-keys -t {session_name} 'mkdir -p {remote_dir}' C-m")


def copy_file(commands, session_name, local_file, account, remote_ip, home_dir):
    return append_command(commands, f"tmux send-keys -t {session_name} 'scp {local_file} {account}@{remote_ip}:{home_dir}' C-m")


def move_file(commands, session_name, source_file, target_dir):
    return append_command(commands, f"tmux send-keys -t {session_name} 'mv {source_file} {target_dir}' C-m")


def extract_tar(commands, session_name, tar_file, remote_dir):
    return append_command(commands, f"tmux send-keys -t {session_name} 'tar -xzf {tar_file} -C {remote_dir}' C-m")


if __name__ == "__main__":
    commands = ""

    account = "..."  # conda env 압축 전송 머신의 계정
    remote_ip = "..."  # conda env 압축 전송 머신의 ip
    home_dir = "~"

    conda_env = "..."  # 압축해서 전송할 conda 환경 이름
    remote_dir = f"~/miniconda3/envs/{conda_env}"
    local_env_file = f"~/{conda_env}.tar.gz"

    session_name = f"..." # 작업을 수행할 tmux-interactive 세션 이름

    # tmux 세션 시작
    commands = start_tmux_session(commands, session_name)

    # Conda 환경을 압축하는 명령어 (conda-pack 사용, tmux 세션 내에서 실행)
    commands = conda_pack(commands, session_name, conda_env, local_env_file)

    # 압축된 Conda 환경 파일 전송 (홈 디렉터리로 일단 전송)
    commands = copy_file(commands, session_name, local_env_file, account, remote_ip, home_dir)

    # SSH 연결
    commands = ssh_connect(commands, session_name, account, remote_ip)

    # 원격 디렉토리 생성
    commands = make_remote_directory(commands, session_name, remote_dir)

    # 원격 서버에서 압축 해제
    commands = extract_tar(commands, session_name, local_env_file, remote_dir)

    # 실제 명령어 실행
    os.system(commands)
    print(f"commands: \n{commands}")
