import subprocess


def deploy_container(local_img, dockerhub_repo, dockerhub_login, tag):
    local = f"{local_img}:{tag}"
    remote = f"{dockerhub_login}/{dockerhub_repo}:{tag}"
    subprocess.call([f"docker", "tag", local, remote])
    subprocess.call(["docker", "push", remote])
