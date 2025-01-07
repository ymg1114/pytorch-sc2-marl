import os
import paramiko


def deploy_ssh_key(remote_ip, username, password, public_key_path):
    # SSH 클라이언트 생성
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # 대상 머신에 SSH 연결
        ssh.connect(remote_ip, username=username, password=password)

        # 공용키 읽기
        with open(public_key_path, 'r') as pubkey_file:
            public_key = pubkey_file.read()

        # 대상 머신의 .ssh 디렉토리 생성
        ssh.exec_command('mkdir -p ~/.ssh')

        # 대상 머신의 authorized_keys 파일에 공용키 추가
        ssh.exec_command(f'echo "{public_key}" >> ~/.ssh/authorized_keys')

        # authorized_keys 파일의 권한 설정
        ssh.exec_command('chmod 600 ~/.ssh/authorized_keys')

        print(f"SSH key deployed to {remote_ip}")

    except Exception as e:
        print(f"Failed to deploy SSH key to {remote_ip}: {e}")

    finally:
        ssh.close()


if __name__ == "__main__":
    remote_ips = ["ip1", "ip2", "ip3"]
    username = "id" # 고정
    password = "pw"  # 고정. 초기 설정 시에만 사용
    public_key_path = os.path.expanduser("~/.ssh/id_rsa.pub") # 일반적인 디폴트 공용키 경로

    for remote_ip in remote_ips:
        deploy_ssh_key(remote_ip, username, password, public_key_path) 