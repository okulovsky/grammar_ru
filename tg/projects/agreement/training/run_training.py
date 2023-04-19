from tg.common import DataBundle
from tg.common.ml.batched_training import IndexedDataBundle
from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training.factories.networks import CtorAdapter
from tg.common.ml.batched_training import context as btc
from tg.common.ml import dft
from yo_fluq_ds import fluq
from tg.common.ml.batched_training.model_handler import BatchedModelHandler
import datetime
from tg.common.ml.training_core import TrainingEnvironment
from tg.grammar_ru.components.yandex_delivery.training_job import TrainingJob
from tg.grammar_ru.components.yandex_storage.s3_yandex_helpers import S3YandexHandler
from tg.grammar_ru.common import Loc
from sklearn.metrics import roc_auc_score
from tg.grammar_ru.components.yandex_delivery.docker_tools import deploy_container
from pathlib import Path
from dotenv import load_dotenv
from tg.projects.agreement.training.deliverable_stuff import TrainingTask
from tg.grammar_ru.common import Loc
from tg.projects.agreement.training.delivery_routine import DeliveryRoutine
import torch
load_dotenv(Loc.root_path / 'environment.env')

EPOCHS = 1#40

project_name = 'agreementproject'
# dataset_name = 'agreement_adj_mid50_0_declination'
# dataset_name = 'agreement_adj_mid50_1_declination'
# dataset_name = 'agreement_adj_tiny_0_declination'
# dataset_name = 'agreement_adj_tiny_all_decl_masked'
dataset_name = 'agreement_adj_mid50_all_decl_masked'
bucket = 'agreementadjbucket'
task_name = f"task_{EPOCHS}ep_{dataset_name}_CE_Smless_Context20_unstratified"


def get_training_job() -> TrainingJob:
    task = TrainingTask()
    task.settings: bt.TrainingSettings
    task.settings.batch_size = 20_000
    task.settings.epoch_count = EPOCHS
    task.optimizer_ctor = CtorAdapter('torch.optim:Adam', ('params',), lr=0.1)
    task.loss_ctor = CtorAdapter(
        "torch.nn:CrossEntropyLoss")

    task.info["dataset"] = dataset_name
    task.info["name"] = task_name

    job = TrainingJob(
        tasks=[task],
        project_name=project_name,
        bucket=bucket
    )
    return job


job = get_training_job()
job.run()
exit()

tag = 'v_' + datetime.datetime.now().time().strftime("%H_%M_%S")
dockerhub_repo = 'agreement'  # name of your repo
dockerhub_login = 'sergio0x0'  # your login

local_img = 'agr_job'  # job name


routine = DeliveryRoutine(job, name=local_img, version=tag)
# routine.local().execute()
routine.build_container()
deploy_container(local_img, dockerhub_repo, dockerhub_login, tag)
