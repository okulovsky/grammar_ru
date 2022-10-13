from contextvars import Context
from typing import *

from enum import Enum
from copy import deepcopy

from ....common.ml import batched_training as bt
from ....common.ml.batched_training import context as btc
from ....common.ml.batched_training import torch as btt
from ....common.ml.batched_training.mirrors import ExtractorNetworkBinding

from .attention_network import AttentionReccurentNetwork, AttentionNetwork
from .bidirect_lstm_network import BidirectLSTMNetwork


class ContextualNetworkType(Enum):
    Plain = 0
    LSTM = 1
    Attention = 2
    AttentionReccurent = 3
    BidirectLSTM = 4


class ContextualBinding(ExtractorNetworkBinding):
    def __init__(self,
                 name: str,
                 context_length: int,
                 network_type: ContextualNetworkType,
                 hidden_size: Union[int, List[int]],
                 context_builder: btc.ContextBuilder,
                 extractor: bt.Extractor,
                 debug=False
                 ):
        super(ContextualBinding, self).__init__(name)
        self.context_length = context_length
        self.network_type = network_type
        self.hidden_size = hidden_size
        self.reverse_order_in_lstm = False
        self.context_builder = deepcopy(context_builder)
        self.extractor = deepcopy(extractor)
        self.debug = debug

    def create_network_factory(self, task, input):
        if self.network_type == ContextualNetworkType.Plain:
            return btt.FullyConnectedNetwork.Factory(self.hidden_size).prepend_extraction(self.name)
        elif self.network_type == ContextualNetworkType.LSTM:
            return btt.LSTMNetwork.Factory(self.hidden_size).prepend_extraction(self.name)
        elif self.network_type == ContextualNetworkType.AttentionReccurent:
            return AttentionReccurentNetwork.Factory(self.hidden_size).prepend_extraction(self.name)
        elif self.network_type == ContextualNetworkType.Attention:
            return AttentionNetwork.Factory().prepend_extraction(self.name)
        elif self.network_type == ContextualNetworkType.BidirectLSTM:
            return BidirectLSTMNetwork.Factory(self.hidden_size).prepend_extraction(self.name)
        else:
            raise ValueError(f"Network type {self.network_type} is not recognized")

    def _create_extractor_internal(self, task, bundle):
        if self.network_type == ContextualNetworkType.Plain:
            eaf = btc.SimpleExtractorToAggregatorFactory(self.extractor, btc.PivotAggregator(True))
            fin = btc.PandasAggregationFinalizer()
        else:
            eaf = btc.SimpleExtractorToAggregatorFactory(self.extractor)
            fin = btt.LSTMFinalizer(self.reverse_order_in_lstm)

        return btc.ContextExtractor(
            self.name,
            self.context_length,
            self.context_builder,
            eaf,
            fin,
            self.debug
        )
