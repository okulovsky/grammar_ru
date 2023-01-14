from typing import *
from ....common.ml import batched_training as bt
from .....common.ml.batched_training.torch.torch_task import Conventions
from .torch_model_handler import TorchModelHandler


class TaskFactory(bt.AbstractTrainingTask):
    def __init__(self):
        super(TaskFactory, self).__init__()
        self.task = None

    def create_task(self, data, env):
        pass

    def instantiate_default_task(self,
                                 epoch_count,
                                 batch_size,
                                 metric_pool,
                                 mini_batch_size=200,
                                 mini_epoch_count=8):
        settings = bt.TrainingSettings(
            epoch_count=epoch_count,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            mini_epoch_count=mini_epoch_count
        )
        self.task = bt.BatchedTrainingTask(
            settings=settings,
            metric_pool=metric_pool,
            splitter=bt.PredefinedSplitter(
                Conventions.SplitColumnName,
                [Conventions.TestName, Conventions.DisplayName],
                [Conventions.TrainName, Conventions.DisplayName]
            )
        )

    def setup_batcher(self, bundle, extractors):
        self.task.batcher = bt.Batcher(self.task.settings.batch_size, extractors)

    def setup_model(self, network_factory, optimizer_ctor='torch.optim:SGD', loss_ctor='torch.nn:MSELoss', learning_rate=0.1):
        self.task.model_handler = TorchModelHandler(network_factory, optimizer_ctor, loss_ctor, learning_rate)


    def run_with_environment(self, data, env=None):
        if self.task is None:
            self.create_task(data, env)
        return self.task.run_with_environment(data, env)

    def get_metric_names(self):
        if self.task is None:
            return []
        return self.task.get_metric_names()










