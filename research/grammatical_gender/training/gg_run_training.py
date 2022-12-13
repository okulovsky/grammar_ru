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

from tg.grammar_ru.common import Loc

load_dotenv(Loc.root_path / 'environment.env')

project_name = 'gg_project'
dataset_name = 'gg_dataset'
bucket = 'ggbucket'


class MulticlassMetrics(bt.Metric):
    def __init__(self, add_accuracy=True, add_rating=False):
        self.add_accuracy = add_accuracy
        self.add_rating = add_rating

    def get_names(self):
        result = []
        if self.add_accuracy:
            result.append('accuracy')
        if self.add_rating:
            result.append('rating')
        return result

    def measure(self, df, _):
        prefix = 'true_label_'
        labels = []
        for c in df.columns:
            if c.startswith(prefix):
                labels.append(c.replace(prefix, ''))

        def ustack(df, prefix, cols, name):
            df = df[[prefix+c for c in cols]]
            df.columns = [c for c in cols]
            df = df.unstack().to_frame(name)
            return df

        predicted = ustack(df, 'predicted_label_', labels, 'predicted')
        true = ustack(df, 'true_label_', labels, 'true')
        df = predicted.merge(true, left_index=True,
                             right_index=True).reset_index()
        df.columns = ['label', 'sample', 'predicted', 'true']
        df = df.feed(fluq.add_ordering_column(
            'sample', ('predicted', False), 'predicted_rating'))

        match = (df.loc[df.predicted_rating ==
                 0].set_index('sample').true > 0.5)
        rating = df.loc[df.true > 0.5].set_index('sample').predicted_rating
        result = []
        if self.add_accuracy:
            result.append(match.mean())
        if self.add_rating:
            result.append(rating.mean())
        return result


def get_multilabel_extractor():
    label_extractor = (bt.PlainExtractor
                       .build(Conventions.LabelFrame)
                       .index()
                       .apply(take_columns=['label'], transformer=dft.DataFrameTransformerFactory.default_factory())
                       )
    return label_extractor


class MyNetworkFactory:
    def __init__(self, nn_head_factory):
        self.nn_head_factory = nn_head_factory

    def create_network(self, task, input):  # input is batch ~ sample
        nn_head = self.nn_head_factory.create_network(task=None, input=input)
        head_out = nn_head(input)
        hidden_size = head_out.shape[1]
        labels_count = input['label'].shape[1]
        nn_tail = FullyConnectedNetwork(
            sizes=[3], input=hidden_size, output=labels_count)
        # return FeedForwardNetwork(nn_head, nn_tail, torch.nn.Softmax(dim=1))
        return FeedForwardNetwork(nn_head, nn_tail)


class ClassificationTask(TaskFactory):
    def get_network(self, _, sample):
        assembled_network_factory = MyNetworkFactory(self.nn_head_factory)
        assembled_network = assembled_network_factory.create_network(
            task=None, input=sample)
        return assembled_network

    def create_task(
        self, data: IndexedDataBundle,
            env: Optional[TrainingEnvironment] = None) -> None:

        metrics = bt.MetricPool().add(MulticlassMetrics())
        self.instantiate_default_task(
            epoch_count=2,
            batch_size=100,
            mini_batch_size=50,
            mini_epoch_count=1,
            metric_pool=metrics
        )

        plain_context_builder = PlainContextBuilder(include_zero_offset=False,
                                                    left_to_right_contexts_proportion=1)

        plain_context = ContextualBinding(
            name='plain_context',
            context_length=3,
            network_type=ContextualNetworkType.Plain,
            hidden_size=[30],
            context_builder=plain_context_builder,
            extractor=CoreExtractor(join_column='another_word_id'),
            debug=False
        )
        self.nn_head_factory = plain_context.create_network_factory(
            task=None, input=None)  # TODO could be better?
        core_extractor = plain_context.create_extractor(task=None, bundle=data)
        self.setup_batcher(data, [core_extractor, get_multilabel_extractor()])
        self.setup_model(self.get_network)


def get_training_job() -> TrainingJob:
    task = ClassificationTask()
    task.info["dataset"] = dataset_name
    task.info["name"] = "gg_task"

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

b_path = Loc.bundles_path/'grammatical_gender/toy'
data = DataBundle.load(b_path)
task = job.tasks[0]
task.name='gg'
model_folder = Path.home() / 'models' / f'{task.name}'
env = FileCacheTrainingEnvironment(model_folder)
# print(data)
success = task.run_with_environment(data, env)

# object_methods = [method_name for method_name in dir(task.task)
#                   if callable(getattr(task.task, method_name))]

# print(object_methods)


# 6a
# routine.local.execute()

# 6b
# build_container(job, 'gg', '1', local_img,
#                 image_tag=tag)
# deploy_container(local_img, dockerhub_repo, dockerhub_login, tag)
