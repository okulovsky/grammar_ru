from tg.grammar_ru.ml.components.yandex_delivery.training_job import TrainingJob
from tg.grammar_ru.ml.components.yandex_storage.s3_yandex_helpers import S3YandexHandler
from tg.common.delivery.jobs.ssh_docker_job_routine import build_container
from tg.common.delivery.jobs import SSHDockerJobRoutine, DockerOptions
from tg.common.delivery.packaging import FakeContainerHandler

from tg.grammar_ru.common import Loc
from sklearn.metrics import roc_auc_score
from typing import Dict, Iterable
from pathlib import Path

from tg.common import DataBundle
from tg.common.ml.batched_training import IndexedDataBundle

from tg.grammar_ru.ml.tasks.style.classification_task import ClassificationTask, Features
from tg.grammar_ru.ml.components import ContextualNetworkType


BUCKET_NAME = "dabdya-bucket"
PROJECT_NAME = "style_books_proza_classification"

DATASETS = {
    Features.GloveEmbedding: Loc.bundles_path/"style/books_proza_glove_3x45k",
    Features.Grammar: Loc.bundles_path/"style/books_proza_grammar_3x45k",
    Features.All: Loc.bundles_path/"style/books_proza_all_3x45k",
}

NETWORKS = {
    ContextualNetworkType.Attention: "attention2d",
    # ContextualNetworkType.AttentionReccurent: "attention3d",
    ContextualNetworkType.LSTM: "lstm"
    # ContextualNetworkType.BidirectLSTM: "bilstm"
}


def get_bundles() -> Iterable[IndexedDataBundle]:
    for ds in DATASETS:
        db = DataBundle.load(ds)
        yield IndexedDataBundle(db.index, db)

def load_environment() -> None:
    import dotenv
    dotenv.load_dotenv(Loc.root_path/"environment.env")

def create_bucket() -> None:
    S3YandexHandler.create_bucket(BUCKET_NAME)

def upload_datasets() -> None:
    s3_path_pattern = "datasphere/{project_name}/datasets/{dataset_name}"
    for _, ds_path in DATASETS.items():
        s3_bundle_path = s3_path_pattern.format(
            project_name = PROJECT_NAME, 
            dataset_name = ds_path.name
        )
        S3YandexHandler.upload_folder(BUCKET_NAME, s3_bundle_path, ds_path)

def get_training_job() -> TrainingJob:
    """Creates bundles_x_networks of training tasks and packs them into job"""
    tasks = []
    for ds_features, ds_path in DATASETS.items():
        for network_type, network_name in NETWORKS.items():
            task = ClassificationTask(network_type, ds_features)
            task.info["name"] = f"{network_name}_{ds_path.name}"
            task.info["dataset"] = ds_path.name
            tasks.append(task)

    job = TrainingJob(
        tasks = tasks,
        project_name = PROJECT_NAME,
        bucket = BUCKET_NAME
    )
    return job

def run_local() -> None:
    job = get_training_job()
    job.run()

def create_docker_container() -> None:
    job = get_training_job()
    build_container(
        job,
        name = job.project_name, version = "v1",
        image_name = PROJECT_NAME, image_tag = "v1"
    )

def login_dockerhub() -> None:
    import subprocess, os, sys
    username = os.environ.get("DOCKER_USER")
    password = os.environ.get("DOCKER_PASS")
    try:
        process = subprocess.run(
            ['docker', 'login', '-u', username, '-p', password], 
            check=True, stdout=sys.stdout, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as err:
        exit_code = err.returncode
        stderror = err.stderr
        print(exit_code, stderror)

def deploy_docker_container() -> None:
    import subprocess, os
    username = os.environ.get("DOCKER_USER")
    docker_repository = os.environ.get("DOCKER_REPO", None)
    local_name = f"{PROJECT_NAME}:v1"
    remote_name = f"{username}/{docker_repository}:{PROJECT_NAME}"
    subprocess.call(["docker", "tag", local_name, remote_name])
    subprocess.call(["docker", "push", remote_name])


if __name__ == "__main__":
    load_environment()
    # create_bucket()
    upload_datasets()
    create_docker_container()
    login_dockerhub()
    deploy_docker_container()
    pass
