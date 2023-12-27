from sklearn.metrics import roc_auc_score

from .. import extractors as ext
from ..models import PunctNetworkEmbedding, PunctNetworkNavec

from ....common.ml.batched_training.factories import CtorAdapter
from ....common.ml import batched_training as bt
from ....common.ml.batched_training import factories as btf


def create_training_task(network_factory, epoch_count=15):
    task = PunctTrainingTask(
        network_factory,
        [
            ext.create_context_extractor(),
            ext.create_label_extractor(),
            ext.create_vocab_extractor(),
            ext.create_navec_extractor(),
        ]
    )
    task.settings.batch_size = 10_000
    task.settings.mini_batch_size = 2000
    task.settings.mini_epoch_count = 5
    task.settings.epoch_count = epoch_count
    task.loss_ctor = CtorAdapter('torch.nn:CrossEntropyLoss')
    task.optimizer_ctor = CtorAdapter('torch.optim:Adam', ('params',))

    return task


class PunctTrainingTask(btf.TorchTrainingTask):
    def __init__(self, factory, extractors):
        super(PunctTrainingTask, self).__init__()
        self.metric_pool = bt.MetricPool().add_sklearn(roc_auc_score)
        self.factory = factory
        self.extractors = extractors

    def initialize_task(self, idb):
        self.setup_batcher(
            idb,
            self.extractors,
            # stratify_by_column='target_word',
            )
        self.setup_model(self.factory, ignore_consistancy_check=True)
