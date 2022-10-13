import torch
from tg.common.ml.batched_training.torch.networks.extracting_network import UniversalFactory


class BidirectLSTMNetwork(torch.nn.Module):
    def __init__(self, input: torch.Tensor, size: int):
        super(BidirectLSTMNetwork, self).__init__()
        self.lstm = torch.nn.LSTM(
            input.shape[2],
            size,
            bidirectional = True
        )      

    def forward(self, input):
        lstm_output = self.lstm(input)
        output = lstm_output[1][0]
        return torch.max(output[0], output[1])

    class Factory(UniversalFactory):
        def __init__(self, size: int):
            super(BidirectLSTMNetwork.Factory, self).__init__(BidirectLSTMNetwork, 'input', size = size)
