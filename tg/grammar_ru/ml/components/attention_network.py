from typing import *
import torch

from tg.common.ml.batched_training.torch.networks.lstm_network import LSTMNetwork
from tg.common.ml.batched_training.torch.networks.extracting_network import UniversalFactory


class AttentionNetwork(LSTMNetwork):
    def __init__(self, input: torch.Tensor, size: int):
        super(AttentionNetwork, self).__init__(input, size)

        n_features = self.lstm.input_size

        self.query, self.key, self.value = [
            torch.nn.Linear(n_features, n_features) for _ in range(3)
        ]

        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, input): # input shape should be (context_length, batch_size, n_features)
        context_length, batch_size, n_features = input.shape
        output = input.clone().detach()
        
        for i in range(batch_size):
            input2d = output[:, i: i+1, :].clone().detach().reshape(context_length, n_features)
            Q, K, V = self.query(input2d), self.key(input2d), self.value(input2d)

            attention_2d = self.softmax( (Q @ K.T) / n_features**0.5) @ V
            output[:, i: i+1, :] = attention_2d.reshape(context_length, 1, n_features)
        
        return super(AttentionNetwork, self).forward(output)

    class Factory(UniversalFactory):
        def __init__(self, size: int):
            super(AttentionNetwork.Factory, self).__init__(AttentionNetwork, 'input', size = size)
