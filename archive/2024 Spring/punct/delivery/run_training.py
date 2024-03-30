import datetime
import functools
import itertools

import numpy as np
from dotenv import load_dotenv

import common

from tg.grammar_ru.components.yandex_delivery.training_job import TrainingJob
from tg.grammar_ru.common import Loc
from tg.grammar_ru.components.yandex_delivery.docker_tools import deploy_container
from tg.projects.agreement.training.delivery_routine import DeliveryRoutine

from tg.projects.punct.delivery.training import PunctTrainingTask, punct_network_factory_navec, create_training_task
from tg.projects.punct import extractors as ext


load_dotenv(Loc.root_path / 'environment.env')

task_name = f"task_{common.datasphere_dataset_name}"


def get_extractors():
    extractors = [
            ext.create_context_extractor(),
            ext.create_label_extractor(),
            ext.create_vocab_extractor(),
            ext.create_navec_extractor(),
    ]

    return extractors


def get_training_parameters():
    dropout_range = np.arange(0.3, 0.6, 0.3)
    embedding_size = np.logspace(4, 5, base=2, num=2)
    hidden_size = np.logspace(5, 7, base=2, num=3)

    for d, e, h in itertools.product(dropout_range, embedding_size, hidden_size):
        yield (d, int(e), int(h))


def get_training_tasks():
    tasks = []
    for d, e, h in get_training_parameters():
        factory = functools.partial(
            punct_network_factory_navec,
            embedding_size=e,
            hidden_size=h,
            out_size=4,
            dropout=d,
        )
        task = create_training_task(
            factory,
            epoch_count=10
        )
        task.info["dataset"] = common.datasphere_dataset_name
        task.info["name"] = task_name + f'_e{e}h{h}d{d}'

        tasks.append(task)

    return tasks


def get_training_job() -> TrainingJob:
    tasks = get_training_tasks()

    job = TrainingJob(
        tasks=tasks,
        project_name=common.project_name,
        bucket=common.datasphere_bucket
    )
    return job


job = get_training_job()
# job.run()
# exit()

tag = 'v_' + datetime.datetime.now().time().strftime("%H_%M_%S")
dockerhub_repo = 'grammar_repo'  # name of your repo
dockerhub_login = 'skypunkaudiowalker'  # your login

local_img = 'punct_job'  # job name


routine = DeliveryRoutine(job, name=local_img, version=tag)
# routine.local().execute()
routine.build_container()
deploy_container(local_img, dockerhub_repo, dockerhub_login, tag)
