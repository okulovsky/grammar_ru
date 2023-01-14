from .....common import DataBundle
from .....common.ml.batched_training import IndexedDataBundle
from .....common.ml import batched_training as bt
from .....common.ml import dft
from ...components.extractor_settings import CoreExtractor
from ...components.plain_context_builder import PlainContextBuilder
from ...components.training_task_factory import TaskFactory
from ...components.training_task_factory import Conventions
from .....common.ml.training_core import TrainingEnvironment
from ...components.contextual_binding import ContextualBinding, ContextualNetworkType
from .....common.ml.batched_training.torch.networks import FeedForwardNetwork, FullyConnectedNetwork
from pathlib import Path
from dotenv import load_dotenv
from yo_fluq_ds import fluq
import datetime
import torch
from typing import Dict, Optional  # TODO delete redundant


from ....common import Loc


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
            sizes=[hidden_size], input=hidden_size, output=labels_count)
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
            epoch_count=120,
            batch_size=30_000,
            mini_batch_size=50,
            mini_epoch_count=1,
            metric_pool=metrics
        )
        # self.task.settings.evaluation_batch_limit = 5
        # self.task.settings.training_batch_limit = 5

        plain_context_builder = PlainContextBuilder(include_zero_offset=False,
                                                    left_to_right_contexts_proportion=1)

        plain_context = ContextualBinding(
            name='plain_context',
            context_length=3,
            network_type=ContextualNetworkType.Plain,
            hidden_size=[],
            context_builder=plain_context_builder,
            extractor=CoreExtractor(join_column='another_word_id'),
            debug=False
        )
        self.nn_head_factory = plain_context.create_network_factory(
            task=None, input=None)  # TODO could be better?
        core_extractor = plain_context.create_extractor(task=None, bundle=data)
        self.setup_batcher(data, [core_extractor, get_multilabel_extractor()])
        self.setup_model(self.get_network, learning_rate=0.1)#,optimizer_ctor='torch.optim:Adam')


class ClassificationTaskByBatchSize(ClassificationTask):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def instantiate_default_task(self, epoch_count, batch_size, metric_pool, mini_batch_size=200, mini_epoch_count=8):
        return super().instantiate_default_task(1, self.batch_size, metric_pool, mini_batch_size, mini_epoch_count)
