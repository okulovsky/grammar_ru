from tg.common import DataBundle
from tg.common.ml.batched_training import IndexedDataBundle

from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training import torch as btt
from tg.common.ml.batched_training import context as btc

from tg.grammar_ru.ml.components.attention_network import AttentionNetwork
from tg.common.ml.batched_training.torch.networks.lstm_network import LSTMFinalizer

from tg.grammar_ru.ml.components.extractor_settings import CoreExtractor
from tg.grammar_ru.ml.components.plain_context_builder import PlainContextBuilder

from tg.grammar_ru.ml.components.training_task_factory import TaskFactory
from tg.grammar_ru.ml.components.training_task_factory import Conventions
from tg.common.ml.training_core import TrainingEnvironment

from tg.grammar_ru.ml.components.yandex_delivery.training_job import TrainingJob
from tg.grammar_ru.ml.components.yandex_storage.s3_yandex_helpers import S3YandexHandler
from tg.common.delivery.jobs import SSHDockerJobRoutine, DockerOptions
from tg.common.delivery.packaging import FakeContainerHandler

from tg.grammar_ru.common import Loc
from sklearn.metrics import roc_auc_score
from typing import Dict, Optional

BUCKET_NAME = "dabdya-bucket"
DATASET_NAME = "martin_big"
TASK_NAME = "books_vs_ficbook_martin"
PROJECT_NAME = "style_grammar_based_classification"
BUNDLE_PATH = Loc.bundles_path/"style/martin_big"


def get_bundle() -> IndexedDataBundle:
    db = DataBundle.load(BUNDLE_PATH)
    return IndexedDataBundle(db.index, db)

def get_label_extractor() -> bt.PlainExtractor:
    label_extractor = (
        bt.PlainExtractor
        .build(Conventions.LabelFrame).index()
        .apply(take_columns=['label'], transformer=None)
    )
    return label_extractor

def build_context_extractor(
    context_length: int, right_proportion: float, 
    aggregator: Optional[btc.ContextAggregator] = None) -> btc.ContextExtractor:
    context_extractor = btc.ContextExtractor(
        name = 'features',
        context_size = context_length,
        context_builder = PlainContextBuilder(True, right_proportion),
        feature_extractor_factory = btc.SimpleExtractorToAggregatorFactory(
            CoreExtractor(join_column='another_word_id')
        ),
        finalizer = LSTMFinalizer(reverse_context_order = False),
        debug = True
    )
    return context_extractor


class ClassificationTask(TaskFactory):
    def get_network(self, _, sample) -> btt.FeedForwardNetwork:
        factory = btt.FeedForwardNetwork.Factory(
            AttentionNetwork.Factory().prepend_extraction("features"),
            btt.FullyConnectedNetwork.Factory(sizes = [], output = 1)
        )
        return factory.create_network(self, sample)

    def create_task(
        self, data: IndexedDataBundle, env: Optional[TrainingEnvironment] = None) -> None:

        metrics = bt.MetricPool().add_sklearn(roc_auc_score)
        self.instantiate_default_task(
            epoch_count = 1, 
            batch_size = 100, 
            mini_batch_size = 50, 
            mini_epoch_count = 2, 
            metric_pool = metrics
        )

        extractors = [
            build_context_extractor(context_length = 25, right_proportion = 1.0), 
            get_label_extractor()
        ]
        self.setup_batcher(data, extractors)
        self.setup_model(self.get_network)


def run_local() -> Dict:
    task = ClassificationTask()
    return task.run(get_bundle())

def save_local(result: Dict) -> None:
    output = result["output"]
    output["result_df"].to_parquet(Loc.temp_path/"result_df.parquet")
    import pickle

    with open(Loc.temp_path/"batcher.pickle", "wb") as f:
        pickle.dump(output["batcher"], f)
    
    with open(Loc.temp_path/"history.pickle", "wb") as f:
        pickle.dump(output["history"], f)

    with open(Loc.temp_path/"network.pickle", "wb") as f:
        pickle.dump(output["model"].network, f)


def load_environment():
    import dotenv
    dotenv.load_dotenv(Loc.root_path/"environment.env")

def create_bucket():
    S3YandexHandler.create_bucket(BUCKET_NAME)

def upload_dataset():
    S3PATH = "datasphere/{project_name}/datasets/{dataset_name}".format(
        project_name = PROJECT_NAME,
        dataset_name = DATASET_NAME
    )
    S3YandexHandler.upload_folder(BUCKET_NAME, S3PATH, BUNDLE_PATH)

def get_training_job() -> TrainingJob:
    task = ClassificationTask()
    task.info["dataset"] = DATASET_NAME
    task.info["name"] = TASK_NAME

    job = TrainingJob(
        tasks = [task],
        project_name = PROJECT_NAME,
        bucket = BUCKET_NAME
    )
    return job

def run_local_with_yandex_object():
    job = get_training_job()
    job.run()

def create_docker_container():
    job = get_training_job()
    routine = SSHDockerJobRoutine(
        job = job,
        remote_host_address=None,
        remote_host_user=None,
        handler_factory = FakeContainerHandler.Factory(),
        options = DockerOptions(propagate_environmental_variables=[])
    )

    routine._name, routine._version = TASK_NAME, "v1"
    routine.local.execute()

def login_dockerhub():
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

def deploy_docker_container():
    import subprocess, os
    username = os.environ.get("DOCKER_USER")
    docker_repository = os.environ.get("DOCKER_REPOSITORY", None)
    local_name = f"{TASK_NAME}:v1"
    remote_name = f"{username}/{docker_repository}:v1"
    subprocess.call(["docker", "tag", local_name, remote_name])
    subprocess.call(["docker", "push", remote_name])


if __name__ == "__main__":
    # result = run_local()
    # save_local(result)

    # load_environment()
    # create_bucket()
    # upload_dataset()
    # run_local_with_yandex_object()
    # create_docker_container()
    # login_dockerhub()
    # deploy_docker_container()
    pass
