import os
import subprocess

from concurrent.futures import ThreadPoolExecutor


def run_command(command, description=""):
    """Run a shell command and handle errors."""
    try:
        print(f"Executing: {command} ({description})")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {description}: {e}")
        raise


def append_command(commands, new_command):
    return commands + new_command + '\n'


def start_tmux_session(commands, session_name):
    return append_command(commands, f"tmux new-session -d -s {session_name}")


def ssh_connect(commands, session_name, account, remote_ip):
    return append_command(commands, f"tmux send-keys -t {session_name} 'ssh {account}@{remote_ip}' C-m")


def make_remote_directory(commands, session_name, remote_dir):
    return append_command(commands, f"tmux send-keys -t {session_name} 'mkdir -p {remote_dir}' C-m")


def copy_file(commands, session_name, local_file, account, remote_ip, home_dir):
    return append_command(commands, f"tmux send-keys -t {session_name} 'scp {local_file} {account}@{remote_ip}:{home_dir}' C-m")


def extract_tar(commands, session_name, tar_file, remote_dir):
    return append_command(commands, f"tmux send-keys -t {session_name} 'tar -xzf {tar_file} -C {remote_dir}' C-m")


def deploy_to_machine(remote_ip, session_name, local_env_file, account, conda_env, home_dir):
    remote_dir = f"~/miniconda3/envs/{conda_env}"
    commands = ""

    # Start tmux session
    commands = start_tmux_session(commands, session_name)

    # Copy packed Conda environment to the remote server
    commands = copy_file(commands, session_name, local_env_file, account, remote_ip, home_dir)

    # SSH into the server
    commands = ssh_connect(commands, session_name, account, remote_ip)

    # Create remote directory
    commands = make_remote_directory(commands, session_name, remote_dir)

    # Extract the tar file on the remote server
    commands = extract_tar(commands, session_name, local_env_file, remote_dir)

    # Execute tmux commands
    run_command(commands, description=f"Deploy to {remote_ip}")
    
    run_command("sleep 300", description=f"Manual Sleep to wait tmux-process to be Done")
    
    # Terminate the tmux session
    run_command(f"tmux kill-session -t {session_name}", description=f"Clean up tmux session {session_name}")
    print(f"Deployment to {remote_ip} completed.")


if __name__ == "__main__":
    home_dir = "~"
    account = "id"  # Fixed account name for SSH
    remote_ips = ["ip1", "ip2", "ip3"]  # List of remote machine IPs
    conda_env = "envName"  # Conda environment to pack and deploy
    local_env_file = f"~/{conda_env}.tar.gz"
    session_names = [f"{ip.replace('.', '_')}_#{idx}" for idx, ip in enumerate(remote_ips)]

    # Pack the local Conda environment
    print("Packing Conda environment...")
    run_command(f"conda pack -n {conda_env} -o {local_env_file}", description="Conda pack")
    print(f"Conda environment packed: {local_env_file}")

    # Deploy the Conda environment to all remote machines in parallel
    print("Starting deployment to remote machines...")
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(deploy_to_machine, ip, session_name, local_env_file, account, conda_env, home_dir)
            for ip, session_name in zip(remote_ips, session_names)
        ]
        for future in futures:
            future.result()

    print("Deployment completed to all machines!")