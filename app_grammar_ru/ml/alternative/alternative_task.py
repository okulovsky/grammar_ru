from typing import *
from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training import torch as btt
from tg.common.ml.batched_training import context as btc
from tg.common.ml.batched_training import gorynych as btg
from grammar_ru import components as cmp
from sklearn.metrics import roc_auc_score
import torch


feature_dict = {
    'p': 'pymorphy',
    'm': 'slovnet_morph',
    's': 'slovnet_syntax',
    'f': 'syntax_fixes',
    't': 'syntax_stats'
}

class AlternativeNetwork(torch.nn.Module):
    def __init__(self, sample, head: torch.nn.Module):
        super().__init__()
        self.head = head
        tensor = self.head(sample)
        self.linear = torch.nn.Linear(tensor.shape[1], 1)

    def forward(self, x):
        x = self.head(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


class AlternativeTrainingTask(btt.TorchTrainingTask):
    def __init__(self,
                 dataset: str,
                 batch_size: int = 100000,
                 epoch_count: int = 100,
                 network_type: btg.Dim3NetworkType = btg.Dim3NetworkType.LSTM,
                 hidden_size: int = 50,
                 tail_size: Optional[int] = None,
                 context_length: int = 15,
                 context_shift: float = 0.5,
                 features: Optional[str] = None,
                 learning_rate: float = 0.1
                 ):
        super(AlternativeTrainingTask, self).__init__()
        self.settings.epoch_count = epoch_count
        self.settings.batch_size = batch_size

        self.info['project_name'] = 'alt'
        self.info['dataset'] = dataset

        self.metric_pool = bt.MetricPool().add_sklearn(roc_auc_score)

        if features is not None:
            features = [feature_dict[c] for c in features]

        self.features = features
        self.factory = btg.Gorynych()



        core_extractor = cmp.CoreExtractor(join_column='another_word_id', allow_list=features)


    def initialize_task(self, idb: bt.IndexedDataBundle):
        label_extractor = (
            bt.PlainExtractor
            .build(btt.Conventions.LabelFrame)
            .index()
            .apply(
                take_columns='label',
                transformer=None
            )
        )

        core_extractor = cmp.CoreExtractor(join_column='another_word_id', allow_list=self.features)
        context_builder = cmp.PlainContextBuilder(include_zero_offset=True, left_to_right_contexts_proportion=0.5)

        self.context_extractor = self.factory.create_context_extractor_from_inner_extractors(
            'context_features',
            [core_extractor],
            context_builder
        )

        extractors = [self.context_extractor, label_extractor]

        self.setup_batcher(idb, extractors)
        self.setup_model(self.create_network, True)


    def create_network(self, batch):
        head = self.factory.create_context_head(batch, self.context_extractor)
        return AlternativeNetwork(batch, head)

