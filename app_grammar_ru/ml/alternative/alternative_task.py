from typing import *
from ...common.ml import batched_training as bt
from ...common.ml.batched_training import factories as btf
from ...common.ml.batched_training import context as btc
from ...grammar_ru import components as cmp
from sklearn.metrics import roc_auc_score

feature_dict = {
    'p': 'pymorphy',
    'm': 'slovnet_morph',
    's': 'slovnet_syntax',
    'f': 'syntax_fixes',
    't': 'syntax_stats'
}

class AlternativeTrainingTask(btf.TorchTrainingTask):
    def __init__(self,
                 dataset: str,
                 batch_size: int = 100000,
                 epoch_count: int = 100,
                 reduction_type: btc.ReductionType = btc.ReductionType.Dim3Folded,
                 network_type: btc.Dim3NetworkType = btc.Dim3NetworkType.LSTM,
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

        core_extractor = cmp.CoreExtractor(join_column='another_word_id', allow_list=features)

        self.context = btc.ContextualAssemblyPoint(
            'features',
            cmp.PlainContextBuilder(
                include_zero_offset=True,
                left_to_right_contexts_proportion=context_shift
            ),
            core_extractor,
            context_length = context_length
        )
        self.context.reduction_type = reduction_type
        self.context.dim_3_network_factory.network_type = network_type
        self.context.hidden_size = hidden_size
        self.optimizer_ctor.type = 'torch.optim:Adam'
        self.optimizer_ctor.kwargs['lr'] = learning_rate
        self.tail_size = tail_size




    def initialize_task(self, idb: bt.IndexedDataBundle):
        label_extractor = (
            bt.PlainExtractor
            .build(btf.Conventions.LabelFrame)
            .index()
            .apply(
                take_columns='label',
                transformer=None
            )
        )
        extractors = [
            self.context.create_extractor(),
            label_extractor
        ]

        self.setup_batcher(idb, extractors)

        head_factory = self.context.create_network_factory()
        factory = btf.FeedForwardNetwork.Factory(
            head_factory,
            btf.Factories.Factory(btf.Perceptron, output_size=self.tail_size),
            btf.Factories.Factory(btf.Perceptron, output_size=1)
        )
        self.setup_model(factory, True)

