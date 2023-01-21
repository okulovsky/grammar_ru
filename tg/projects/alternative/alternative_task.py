from ...common.ml import batched_training as bt
from ...common.ml.batched_training import factories as btf
from ...common.ml.batched_training import context as btc
from ...grammar_ru import components as cmp
from sklearn.metrics import roc_auc_score


class AlternativeTrainingTask(btf.TorchTrainingTask):
    def __init__(self):
        super(AlternativeTrainingTask, self).__init__()
        self.metric_pool = bt.MetricPool().add_sklearn(roc_auc_score)
        core_extractor = cmp.CoreExtractor(join_column='another_word_id')

        self.context = btc.ContextualAssemblyPoint(
            'features',
            cmp.PlainContextBuilder(
                include_zero_offset=True,
                left_to_right_contexts_proportion=0.5
            ),
            core_extractor,
        )
        self.optimizer_ctor.type = 'torch.optim:Adam'

        self.tail_size = 50

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