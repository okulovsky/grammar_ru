from unittest import TestCase
from tg.common.ml.batched_training import sandbox as bts
from tg.common.ml.batched_training import factories as btf
from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training import InMemoryTrainingEnvironment
from sklearn.metrics import roc_auc_score
import torch


def get_bundle_and_task():
    bundle = bts.get_binary_classification_bundle()
    task = bts.SandboxTorchTask(
        [
            bts.get_feature_extractor(),
            bts.get_binary_label_extractor()
        ],
        (100,),
        roc_auc_score
    )
    task.settings.epoch_count = 10
    return bundle, task


class NetworkFactory:
    def __init__(self, softmaxless_network_factory):
        self.softmaxless_network_factory = softmaxless_network_factory

    def __call__(self, batch):
        softmaxless_net = self.softmaxless_network_factory(batch)
        return torch.nn.Sequential(softmaxless_net, torch.nn.Softmax(dim=1))


class SoftmaxTorchTask(bts.SandboxTorchTask):

    def initialize_task(self, data):
        self.setup_batcher(data, self.extractors)
        softmaxless_network_factory = btf.Factories.Tailing(
            btf.Factories.FullyConnected(self.network_sizes, self.input_frame_name),
            btf.Conventions.LabelFrame
        )
        softmaxed_factory = NetworkFactory(softmaxless_network_factory)
        self.setup_model(softmaxed_factory, ignore_consistancy_check=True)


class PytorchCrossEntropyTestCase(TestCase):
    def test_multiclass_classification_with_cross_entropy_and_softmax(self):
        bundle = bts.get_multilabel_classification_bundle()
        task = SoftmaxTorchTask(
            [
                bts.get_feature_extractor(),
                bts.get_multilabel_extractor()
            ],
            (30,),
            bt.MulticlassMetrics(True, True, [1])
        )
        task.settings.epoch_count = 10
        task.loss_ctor = btf.CtorAdapter("torch.nn:CrossEntropyLoss")
        task.optimizer_ctor = btf.CtorAdapter('torch.optim:Adam', ('params',), lr=0.1)
        env = InMemoryTrainingEnvironment()
        task.run_with_environment(bundle, env)
        result = env.result
        self.assertGreater(result['metrics']['recall_at_1_test'], 0.6)
