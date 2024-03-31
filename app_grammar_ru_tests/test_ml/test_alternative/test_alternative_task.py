from tg.common.ml.batched_training import gorynych as btg
from app_grammar_ru.ml.alternative import AlternativeTrainingTask
from unittest import TestCase
from pathlib import Path


class AlternativeTestCase(TestCase):
    def run_test(self, network_type):
        task = AlternativeTrainingTask('test')
        task.settings.epoch_count = 1
        task.settings.batch_size = 20000
        task.settings.mini_epoch_count = 10
        task.optimizer_ctor.type='torch.optim:Adam'
        task.factory.context_dimensionality_reduction_type = network_type
        result = task.run(Path(__file__).parent/'tsa-test.zip')
        self.assertGreater(result['metrics']['roc_auc_score_test'], 0.7)

    def test_alon_attention(self):
        self.run_test(btg.Dim3NetworkType.AlonAttention)

