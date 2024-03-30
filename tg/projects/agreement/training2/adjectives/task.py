from pathlib import Path
from typing import List, Union

import joblib
import pandas as pd
import torch

from tg.common.ml import batched_training as bt
from tg.common.ml import dft
from tg.common.ml.batched_training import context as btc
from tg.common.ml.batched_training import factories as btf
from tg.common.ml.batched_training.factories import TorchModelHandler, CtorAdapter
from tg.grammar_ru.components import CoreExtractor, PlainContextBuilder
from tg.projects.agreement.training2.adjectives.common import (
    MulticlassPredictionInterpreter,
)
from tg.projects.agreement.training2.adjectives.metrics import (
    AlternativeTaskMulticlassMetrics,
)


def create_assembly_point(context_length=6):
    context_builder = PlainContextBuilder(
        include_zero_offset=False, left_to_right_contexts_proportion=0.5
    )

    ap = btc.ContextualAssemblyPoint(
        name="features",
        context_builder=context_builder,
        extractor=CoreExtractor(join_column="another_word_id"),
        context_length=context_length,
    )
    ap.reduction_type = ap.reduction_type.Dim3Folded
    return ap


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
            f"Argument {argument_name} is supposed to be int, Tensor or none, but was `{argument}`"
        )


class FullyConnectedNetwork(torch.nn.Module):
    def __init__(
        self,
        sizes: List[int],
        input: Union[None, torch.Tensor, int] = None,
        output: Union[None, torch.Tensor, int] = None,
    ):
        super(FullyConnectedNetwork, self).__init__()
        sizes = _update_sizes_with_argument("input", input, sizes, lambda s, v: [v] + s)
        sizes = _update_sizes_with_argument(
            "output", output, sizes, lambda s, v: s + [v]
        )
        self.layers = torch.nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, input):
        X = input
        for layer in self.layers:
            X = layer(X)
        return X


class Network(torch.nn.Module):
    def __init__(self, head, hidden_size, batch):
        super(Network, self).__init__()
        self.head = head
        self.tail = FullyConnectedNetwork(
            sizes=[], input=hidden_size, output=batch.index_frame.label.nunique()
        )

    def forward(self, batch):
        return self.tail(self.head(batch))


class NetworkFactory:
    def __init__(self, assembly_point):
        self.assembly_point = assembly_point

    def __call__(self, batch):
        head_factory = self.assembly_point.create_network_factory()
        head = head_factory(batch)
        return Network(head, self.assembly_point.hidden_size, batch)


def get_multilabel_extractor():
    label_extractor = (
        bt.PlainExtractor.build(btf.Conventions.LabelFrame)
        .index()
        .apply(
            take_columns=["label"],
            transformer=dft.DataFrameTransformerFactory.default_factory(
                max_values_per_category=1000
            ),
        )
    )
    return label_extractor


class AdjectivesTrainingTask(btf.TorchTrainingTask):
    def __init__(
        self,
        *,
        assembly_point: btc.ContextualAssemblyPoint,
        network_factory: NetworkFactory,
        label_count: int,
    ):
        super(AdjectivesTrainingTask, self).__init__()
        self.metric_pool = bt.MetricPool().add(
            AlternativeTaskMulticlassMetrics(label_count=label_count)
        )
        self.assembly_point = assembly_point
        self.network_factory = network_factory

    def initialize_task(self, idb):
        self.setup_batcher(
            idb, [self.assembly_point.create_extractor(), get_multilabel_extractor()]
        )
        self.setup_model(self.network_factory, ignore_consistency_check=True)

    def setup_model(self, network_factory, ignore_consistency_check=False):
        self.model_handler = TorchModelHandler(
            network_factory,
            self.optimizer_ctor,
            self.loss_ctor,
            ignore_consistency_check,
        )
        self.model_handler.multiclass_prediction_interpreter = (
            MulticlassPredictionInterpreter()
        )


def create_empty_training_task(
    *,
    label_count: int,
    context_length: int = 15,
    hidden_size: int = 50,
    epoch_count: int = 5,
) -> AdjectivesTrainingTask:
    ap = create_assembly_point(context_length=context_length)
    ap.hidden_size = hidden_size
    ap.dim_3_network_factory.network_type = btc.Dim3NetworkType.LSTM
    network_factory = NetworkFactory(ap)

    task = AdjectivesTrainingTask(
        assembly_point=ap, network_factory=network_factory, label_count=label_count
    )
    task.settings.epoch_count = epoch_count
    task.optimizer_ctor = CtorAdapter("torch.optim:Adam", ("params",), lr=3e-4)
    task.loss_ctor = CtorAdapter("torch.nn:CrossEntropyLoss")

    return task


def train_adjectives_model(
    idb: bt.IndexedDataBundle, save_path: Path, **task_kwargs
) -> None:
    label_count = idb.index_frame.label.nunique()
    task = create_empty_training_task(label_count=label_count, **task_kwargs)
    result = task.run(idb)

    trained_task = result["output"]["training_task"]
    joblib.dump(trained_task, save_path)


def load_trained_task(task_path: Path) -> AdjectivesTrainingTask:
    return joblib.load(task_path)
