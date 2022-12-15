from typing import *
import torch

from .attention_layer import Attention3D, Attention2D
from tg.common.ml.batched_training.torch.networks.extracting_network import UniversalFactory


class AttentionReccurentNetwork(torch.nn.Module):
    def __init__(self, input: torch.Tensor, size: int):
        super(AttentionReccurentNetwork, self).__init__()

        n_features = input.shape[2]

        self.model = torch.nn.Sequential(
            Attention3D(n_features),
            torch.nn.LSTM(n_features, size)
        )

    def forward(self, input):
        lstm_output = self.model(input)
        output = lstm_output[1][0]
        output = output.reshape(output.shape[1], output.shape[2])
        return output

    class Factory(UniversalFactory):
        def __init__(self, size: int):
            super(AttentionReccurentNetwork.Factory, self).__init__(
                AttentionReccurentNetwork, 'input', size = size
            )


class AttentionNetwork(torch.nn.Module):
    def __init__(self, input: torch.Tensor):
        super(AttentionNetwork, self).__init__()

        n_features = input.shape[2]

        self.model = torch.nn.Sequential(
            Attention2D(n_features)
        )

    def forward(self, input):
        return self.model(input)

    class Factory(UniversalFactory):
        def __init__(self):
            super(AttentionNetwork.Factory, self).__init__(AttentionNetwork, 'input')
