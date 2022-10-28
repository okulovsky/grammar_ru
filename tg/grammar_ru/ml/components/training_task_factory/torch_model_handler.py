from typing import *
from .....common.ml import batched_training as bt
from .....common.ml.batched_training.torch import networks as btn
from .....common.ml.single_frame_training import ModelConstructor

import pandas as pd
from .conventions import Conventions
import torch

class OptimizerConstructor:
    def __init__(self, type_name, **kwargs):
        self.type_name = type_name
        self.kwargs = kwargs

    def instantiate(self, params):
        cls = ModelConstructor._load_class(self.type_name)
        return cls(params, **self.kwargs)


class TorchModelHandler(bt.BatchedModelHandler):
    def __init__(self,
                 network_factory,
                 optimizer_ctor,
                 loss_ctor,
                 learning_rate = 0.1
                 ):
        if callable(optimizer_ctor):
            self.optimizer_ctor = optimizer_ctor
        elif isinstance(optimizer_ctor, str):
            self.optimizer_ctor = OptimizerConstructor(optimizer_ctor, lr=learning_rate)
        else:
            raise ValueError(f'optimizer_ctor should be str or callable, but was {type(optimizer_ctor)}')

        if callable(loss_ctor):
            self.loss_ctor = loss_ctor
        elif isinstance(loss_ctor, str):
            self.loss_ctor = ModelConstructor(loss_ctor)
        else:
            raise ValueError(f'optimizer_ctor should be str or callable, but was {type(optimizer_ctor)}')

        self.network_factory = network_factory

    def instantiate(self, task, input: Dict[str, pd.DataFrame]) -> None:
        if isinstance(self.network_factory, btn.TorchNetworkFactory):
            self.network_factory.preview_batch(input)
            self.network = self.network_factory.create_network(task, input)
        else:
            self.network = self.network_factory(task, input)
        self.optimizer = self.optimizer_ctor.instantiate(self.network.parameters())
        self.loss = self.loss_ctor()

    def _predict_1_dim(self, input, labels):
        output = self.network(input)
        output = output.flatten().tolist()
        result = input['index'].copy()
        result['true'] = labels[labels.columns[0]]
        result['predicted'] = output
        return result

    def _predict_multi_dim(self, input, labels):
        result = input['index'].copy()
        output = self.network(input)
        for i, c in enumerate(labels.columns):
            result['true_' + c] = labels[c]
            result['predicted_' + c] = output[:, i].tolist()
        return result

    def predict(self, input: Dict[str, pd.DataFrame]):
        labels = input[Conventions.LabelFrame]
        if labels.shape[1] == 1:
            return self._predict_1_dim(input, labels)
        else:
            return self._predict_multi_dim(input, labels)

    def _train_1_dim(self, input, labels):
        self.optimizer.zero_grad()
        result = self.network(input).flatten()
        target = torch.tensor(input[Conventions.LabelFrame].values).float().flatten()
        loss = self.loss(result, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_multi_dim(self, input, labels):
        self.optimizer.zero_grad()
        result = self.network(input)
        target = torch.tensor(input[Conventions.LabelFrame].values).float()
        loss = self.loss(result, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, input: Dict[str, pd.DataFrame]) -> float:
        labels = input[Conventions.LabelFrame]
        if labels.shape[1] == 1:
            return self._train_1_dim(input, labels)
        else:
            return self._train_multi_dim(input, labels)
