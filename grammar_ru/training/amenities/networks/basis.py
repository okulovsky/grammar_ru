
from typing import *
import pandas as pd
import torch
from tg.common.ml import dft
from tg.common.ml import  batched_training as bt
from functools import partial

def df_to_torch(df):
    return torch.tensor(df.astype(float).values).float()

def input_to_torch(input: Dict[str, pd.DataFrame], *frames: str):
    tensors = []
    for input_name in frames:
        tensor = df_to_torch(input[input_name])
        tensors.append(tensor)
    tensor = torch.cat(tensors, 1)
    return tensor


class LayeredNetwork(torch.nn.Module):
    def __init__(self, sample_tensor, sizes):
        super(LayeredNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        sizes = [sample_tensor.shape[1]] + sizes
        for i in range(len(sizes) - 1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, input):
        X = input
        for layer in self.layers:
            # print(X.shape, layer)
            X = layer(X)
            X = torch.sigmoid(X)
        return X


class DirectInputNetwork(torch.nn.Module):
    def __init__(self, task, input, input_frame):
        super(DirectInputNetwork, self).__init__()
        self.input_frame = input_frame

    def forward(self, input: Dict[str, pd.DataFrame]):
        tensor = input_to_torch(input, self.input_frame)
        return tensor


class LayeredNetworkWithExtraction(torch.nn.Module):
    def __init__(self, task, input, source_frame: List[str], dst_frame, size):
        super(LayeredNetworkWithExtraction, self).__init__()
        if isinstance(source_frame, str):
            source_frame = [source_frame]
        self.source_frame = source_frame
        self.dst_frame = dst_frame
        df = input_to_torch(input,*self.source_frame)
        target_size = input[self.dst_frame].shape[1]
        self.network = LayeredNetwork(df, size+[target_size])

    def forward(self, input: Dict[str, pd.DataFrame]):
        return self.network(input_to_torch(input, *self.source_frame))


def create_feature_transformer():
    transformer = (dft.DataFrameTransformerFactory()
         .on_continuous(dft.ContinousTransformer)
         .on_categorical(partial(
             dft.CategoricalTransformer,
             postprocessor=dft.OneHotEncoderForDataframe()))
         .on_rich_category(25, partial(
             dft.CategoricalTransformer,
             postprocessor=dft.OneHotEncoderForDataframe(),
             replacement_strategy = dft.TopKPopularStrategy(25,'OTHER')
        )))
    return transformer



class TorchModelHandler(bt.BatchedModelHandler):
    def __init__(self, network_factory, learning_rate):
        self.network_factory = network_factory
        self.learning_rate = learning_rate


    def instantiate(self, task, input: Dict[str,pd.DataFrame]) -> None:
        self.network = self.network_factory(task, input)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)
        self.loss = torch.nn.MSELoss()

    def train(self, input: Dict[str, pd.DataFrame]) -> float:
        self.optimizer.zero_grad()
        result = self.network(input)
        target = torch.tensor(input['labels'].values).float()
        loss = self.loss(result, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def predict(self, input: Dict[str, pd.DataFrame]):
        output = self.network(input)
        output = output.flatten().tolist()
        result = input['index'].copy()
        result['true'] = input['labels'].values
        result['predicted'] = output
        return result