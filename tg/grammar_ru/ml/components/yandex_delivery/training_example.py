from sklearn import datasets
import pandas as pd
from .....common.ml import batched_training as bt
import torch

from tg.common.ml import dft

from sklearn.metrics import roc_auc_score

from .....common import DataBundle
from .....common.delivery.training.architecture import FileCacheTrainingEnvironment
from .....common import Logger
from .....common.delivery.jobs import DeliverableJob
from ....common.loc import Loc
from ...components.training_task_factory import TaskFactory, Conventions
from ..yandex_storage.s3_yandex_helpers import S3YandexHandler
from ....ml.components.yandex_delivery.training_job import TrainingJob

from yo_fluq_ds import *

Logger.disable()



def get_feature_extractor():
    feature_extractor = (bt.PlainExtractor
                         .build('features')
                         .index('features')
                         .apply(transformer=dft.DataFrameTransformerFactory.default_factory())
                         )
    return feature_extractor


def get_multilabel_extractor():
    label_extractor = (bt.PlainExtractor
                       .build(Conventions.LabelFrame)
                       .index()
                       .apply(take_columns=['label'], transformer=dft.DataFrameTransformerFactory.default_factory())
                       )
    return label_extractor



class ClassificationNetwork(torch.nn.Module):
    def __init__(self, hidden_size, sample):
        super(ClassificationNetwork, self).__init__()
        self.hidden = torch.nn.Linear(sample['features'].shape[1], hidden_size)
        self.output = torch.nn.Linear(hidden_size, sample['label'].shape[1])

    def forward(self, input):
        X = input['features']
        X = torch.tensor(X.astype(float).values).float()
        X = self.hidden(X)
        X = torch.sigmoid(X)
        X = self.output(X)
        X = torch.sigmoid(X)
        return X


class MulticlassMetrics(bt.Metric):
    def __init__(self, add_accuracy=True, add_rating=False):
        self.add_accuracy = add_accuracy
        self.add_rating = add_rating

    def get_names(self):
        result = []
        if self.add_accuracy:
            result.append('accuracy')
        if self.add_rating:
            result.append('rating')
        return result

    def measure(self, df, _):
        prefix = 'true_label_'
        labels = []
        for c in df.columns:
            if c.startswith(prefix):
                labels.append(c.replace(prefix, ''))

        def ustack(df, prefix, cols, name):
            df = df[[prefix+c for c in cols]]
            df.columns = [c for c in cols]
            df = df.unstack().to_frame(name)
            return df

        predicted = ustack(df, 'predicted_label_', labels, 'predicted')
        true = ustack(df, 'true_label_', labels, 'true')
        df = predicted.merge(true, left_index=True,
                             right_index=True).reset_index()
        df.columns = ['label', 'sample', 'predicted', 'true']
        df = df.feed(fluq.add_ordering_column(
            'sample', ('predicted', False), 'predicted_rating'))

        match = (df.loc[df.predicted_rating ==
                 0].set_index('sample').true > 0.5)
        rating = df.loc[df.true > 0.5].set_index('sample').predicted_rating
        result = []
        if self.add_accuracy:
            result.append(match.mean())
        if self.add_rating:
            result.append(rating.mean())
        return result


def _inner(x, sample):
    return ClassificationNetwork(20, sample)


class ClassificationTask(TaskFactory):
    def create_task(self, data, env):
        metrics = bt.MetricPool().add(MulticlassMetrics())
        self.instantiate_default_task(
            epoch_count=20, batch_size=10000, mini_batch_size=None, metric_pool=metrics)
        self.setup_batcher(
            data, [get_feature_extractor(), get_multilabel_extractor()])
        self.setup_model(_inner, learning_rate=1)




    
