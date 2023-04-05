from typing import List, Union
from pathlib import Path
from dotenv import load_dotenv
from yo_fluq_ds import fluq
import torch
import pandas as pd

from ....common.ml.training_core import TrainingEnvironment
from ....common.ml.batched_training import IndexedDataBundle
from ....common.ml import batched_training as bt
from ....common.ml import dft
from ....grammar_ru.components import CoreExtractor, PlainContextBuilder
from ....common.ml.batched_training.factories import Conventions, FeedForwardNetwork
from ....common.ml.batched_training.context import ContextualAssemblyPoint
from ....common.ml.batched_training import factories as btf
from ....common.ml import batched_training as bt
from ....common.ml import dft
from ....common import Loc
from ....common.ml import batched_training as bt
from ....common.ml.batched_training import context as btc
from ....grammar_ru.components import CoreExtractor


def create_assembly_point(context_length=6):
    ap = btc.ContextualAssemblyPoint(
        name='features',
        context_builder=PlainContextBuilder(
            include_zero_offset=True,
            left_to_right_contexts_proportion=0.5
        ),
        extractor=CoreExtractor(join_column='another_word_id'),
        context_length=context_length
    )
    ap.reduction_type = ap.reduction_type.Dim3Folded
    return ap


def get_multilabel_extractor():
    label_extractor = (bt.PlainExtractor
                       .build(btf.Conventions.LabelFrame)
                       .index()
                       .apply(take_columns=['label'],
                              transformer=dft.DataFrameTransformerFactory.default_factory())
                       )
    return label_extractor


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


def _update_sizes_with_argument(argument_name, argument, sizes, modificator):
    if argument is None:
        return sizes
    elif isinstance(argument, torch.Tensor):
        return modificator(sizes, argument.shape[1])
    elif isinstance(argument, pd.DataFrame):
        return modificator(sizes, argument.shape[1])
    elif isinstance(argument, int):
        return modificator(sizes, argument)
    else:
        raise ValueError(
            f"Argument {argument_name} is supposed to be int, Tensor or none, but was `{argument}`")


class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self,
                 sizes: List[int],
                 input: Union[None, torch.Tensor, int] = None,
                 output: Union[None, torch.Tensor, int] = None):
        super(FullyConnectedNetwork, self).__init__()
        sizes = _update_sizes_with_argument(
            'input', input, sizes, lambda s, v: [v] + s)
        sizes = _update_sizes_with_argument(
            'output', output, sizes, lambda s, v: s + [v])
        self.layers = torch.nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, input):
        X = input
        for layer in self.layers:
            X = layer(X)
            X = torch.sigmoid(X)
        return X


class Network(torch.nn.Module):
    def __init__(self, head, hidden_size, batch):
        super(Network, self).__init__()
        self.head = head
        self.tail = FullyConnectedNetwork(
            sizes=[], input=hidden_size, output=batch.index_frame.label.nunique())
        self.sm = torch.nn.Softmax(dim=1)
        # TODO:relu?

    def forward(self, batch):
        return (
            self.sm(
                self.tail(
                    self.head(batch)))
                    )


class NetworkFactory:
    def __init__(self, assembly_point):
        self.assembly_point = assembly_point

    def __call__(self, batch):
        head_factory = self.assembly_point.create_network_factory()
        head = head_factory(batch)
        return Network(head, self.assembly_point.hidden_size,  batch)


class TrainingTask(btf.TorchTrainingTask):
    def __init__(self):
        super(TrainingTask, self).__init__()
        self.metric_pool = bt.MetricPool().add(MulticlassMetrics())
        self.features_ap = create_assembly_point()

    def initialize_task(self, idb):
        ap = create_assembly_point()
        ap.hidden_size = 50
        ap.dim_3_network_factory.network_type = btc.Dim3NetworkType.LSTM
        # head_factory = ap.create_network_factory()
        self.setup_batcher(
            idb, [ap.create_extractor(), get_multilabel_extractor()])
        self.setup_model(NetworkFactory(ap), ignore_consistancy_check=True)
