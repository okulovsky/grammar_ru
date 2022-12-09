from .....common import DataBundle
from .....common.ml.batched_training import IndexedDataBundle

from .....common.ml import batched_training as bt
from .....common.ml.batched_training import torch as btt
from .....common.ml.batched_training import context as btc

from .....grammar_ru.ml.components.attention_network import AttentionNetwork
from .....common.ml.batched_training.torch.networks.lstm_network import LSTMFinalizer

from .....grammar_ru.ml.components.extractor_settings import CoreExtractor
from .....grammar_ru.ml.components.plain_context_builder import PlainContextBuilder

from .....grammar_ru.ml.components.training_task_factory import TaskFactory
from .....grammar_ru.ml.components.training_task_factory import Conventions
from .....common.ml.training_core import TrainingEnvironment

from .....grammar_ru.ml.components import ContextualNetworkType

from .....grammar_ru.common import Loc
from sklearn.metrics import roc_auc_score
from typing import Optional

from enum import Enum

class Features(Enum):
    All = 0
    Grammar = 1
    GloveEmbedding = 2

def get_label_extractor() -> bt.PlainExtractor:
    label_extractor = (
        bt.PlainExtractor
        .build(Conventions.LabelFrame).index()
        .apply(take_columns=['label'], transformer=None)
    )
    return label_extractor

def build_context_extractor(
    context_length: int, right_proportion: float, features: Features,
    aggregator: Optional[btc.ContextAggregator] = None) -> btc.ContextExtractor:

    grammar_extractor = CoreExtractor(join_column="another_word_id")
    glove_extractor = (
        bt.PlainExtractor.build(name = "glove_features").index()
        .join(frame_name="glove_keys", on_columns="another_word_id")
        .join(frame_name="glove_scores", on_columns="glove_index")
        .apply(
            raise_if_rows_are_missing = False, 
            raise_if_nulls_detected = False, 
            coalesce_nulls = 0, 
            transformer = None
        )
    )

    extractors = [grammar_extractor, glove_extractor]
    if features == Features.Grammar:
        extractors = [grammar_extractor]
    elif features == Features.GloveEmbedding:
        extractors = [glove_extractor]

    context_extractor = btc.ContextExtractor(
        name = 'features',
        context_size = context_length,
        context_builder = PlainContextBuilder(True, right_proportion),
        feature_extractor_factory = btc.SimpleExtractorToAggregatorFactory(
            bt.CombinedExtractor(name = "features", extractors = extractors)
        ),
        finalizer = LSTMFinalizer(reverse_context_order = False),
        debug = True
    )
    return context_extractor


class ClassificationTask(TaskFactory):
    def __init__(
        self, network_type: ContextualNetworkType, features: Features, *args, **kwargs) -> None:
        self.network_type = network_type
        self.features = features
        super(ClassificationTask, self).__init__(*args, **kwargs)

    def assemble_network_factory(self, hidden_size, name):
        if self.network_type == ContextualNetworkType.Plain:
            return btt.FullyConnectedNetwork.Factory(hidden_size).prepend_extraction(name)
        elif self.network_type == ContextualNetworkType.LSTM:
            return btt.LSTMNetwork.Factory(hidden_size).prepend_extraction(name)
        elif self.network_type == ContextualNetworkType.AttentionReccurent:
            return AttentionReccurentNetwork.Factory(hidden_size).prepend_extraction(name)
        elif self.network_type == ContextualNetworkType.Attention:
            return AttentionNetwork.Factory().prepend_extraction(name)
        elif self.network_type == ContextualNetworkType.BidirectLSTM:
            return BidirectLSTMNetwork.Factory(hidden_size).prepend_extraction(name)
        else:
            raise ValueError(f"Network type {self.network_type} is not recognized")

    def get_network(self, _, sample) -> btt.FeedForwardNetwork:
        factory = btt.FeedForwardNetwork.Factory(
            self.assemble_network_factory(20, "features"),
            btt.FullyConnectedNetwork.Factory(sizes = [], output = 1)
        )
        return factory.create_network(self, sample)

    def create_task(
        self, data: IndexedDataBundle, env: Optional[TrainingEnvironment] = None) -> None:

        metrics = bt.MetricPool().add_sklearn(roc_auc_score)
        self.instantiate_default_task(
            epoch_count = 50, 
            batch_size = 10000, 
            mini_batch_size = 200, 
            mini_epoch_count = 4, 
            metric_pool = metrics
        )

        extractors = [
            build_context_extractor(
                context_length = 25, right_proportion = 1.0, features = self.features
            ), 
            get_label_extractor()
        ]
        self.setup_batcher(data, extractors)
        self.setup_model(self.get_network)
