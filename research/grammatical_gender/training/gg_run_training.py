from tg.common import DataBundle
from tg.common.ml.batched_training import IndexedDataBundle

from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training import torch as btt
from tg.common.ml.batched_training import context as btc
from tg.common.ml import dft
from yo_fluq_ds import fluq


from tg.grammar_ru.ml.components.attention_network import AttentionNetwork
from tg.common.ml.batched_training.torch.networks.lstm_network import LSTMFinalizer

from tg.grammar_ru.ml.components.extractor_settings import CoreExtractor
from tg.grammar_ru.ml.components.plain_context_builder import PlainContextBuilder
import datetime
from tg.grammar_ru.ml.components.training_task_factory import TaskFactory
from tg.grammar_ru.ml.components.training_task_factory import Conventions
from tg.common.ml.training_core import TrainingEnvironment
from tg.grammar_ru.ml.components.contextual_binding import ContextualBinding, ContextualNetworkType

from tg.grammar_ru.ml.components.yandex_delivery.training_job import TrainingJob
from tg.grammar_ru.ml.components.yandex_storage.s3_yandex_helpers import S3YandexHandler
from tg.common.delivery.jobs import SSHDockerJobRoutine, DockerOptions
from tg.common.delivery.packaging import FakeContainerHandler

from tg.grammar_ru.common import Loc
from sklearn.metrics import roc_auc_score
from typing import Dict, Optional  # TODO delete redundant
from tg.common.delivery.training.architecture import FileCacheTrainingEnvironment
from tg.grammar_ru.ml.components.yandex_delivery.docker_tools import deploy_container
from tg.common.delivery.jobs.ssh_docker_job_routine import build_container
from tg.common.ml.batched_training.torch.networks import FeedForwardNetwork, FullyConnectedNetwork
import torch
from pathlib import Path
from dotenv import load_dotenv
from tg.grammar_ru.ml.tasks.grammatical_gender.deliverable_stuff import ClassificationTask
from tg.grammar_ru.common import Loc

load_dotenv(Loc.root_path / 'environment.env')

project_name = 'gg_project'
dataset_name = 'gg_lenta_big'
bucket = 'ggbucket'


def get_training_job() -> TrainingJob:
    task = ClassificationTask()
    task.info["dataset"] = dataset_name
    task.info["name"] = "gg_task_lenta_big_20K_100ep"

    job = TrainingJob(
        tasks=[task],
        project_name=project_name,
        bucket=bucket
    )
    return job


job = get_training_job()

routine = SSHDockerJobRoutine(
    job=job,
    remote_host_address=None,
    remote_host_user=None,
    handler_factory=FakeContainerHandler.Factory(),
    options=DockerOptions(propagate_environmental_variables=["AWS_ACCESS_KEY_ID",
                                                             "AWS_SECRET_ACCESS_KEY"])
)
tag = 'v_' + datetime.datetime.now().time().strftime("%H_%M_%S")
dockerhub_repo = 'grammar_repo'  # name of your repo
dockerhub_login = 'sergio0x0'  # your login

local_img = 'gg_img'


# job.run()
# b_path = Loc.bundles_path/'grammatical_gender/toy'
# data = DataBundle.load(b_path)

# task = ClassificationTask()
# task.create_task(data)
# temp_batch = task.task.generate_sample_batch(data,0)


# task = job.tasks[0]

# task.name = 'gg'
# model_folder = Path.home() / 'models' / f'{task.name}'
# env = FileCacheTrainingEnvironment(model_folder)
# print(data)
# success = task.run_with_environment(data, env)
# task.in
# object_methods = [method_name for method_name in dir(task.task)
#                   if callable(getattr(task.task, method_name))]

# print(object_methods)

# 6a
# routine.local.execute()

# 6b
build_container(job, 'gg', '1', local_img,
                image_tag=tag)
deploy_container(local_img, dockerhub_repo, dockerhub_login, tag)
print(tag)