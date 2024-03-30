import torch
from torch import nn
from navec import Navec
from slovnet.model.emb import NavecEmbedding

from ...common.ml.batched_training import context as btc
from ...common.ml import batched_training as bt
from ...grammar_ru.common import Loc


class PunctNetworkEmbedding(nn.Module):
    def __init__(self, embedding_size, features_size, hidden_size, out_size, dropout=0, num_embeddings=10_001):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_size,
        )
        self.lstm = btc.LSTMNetwork(embedding_size + features_size, hidden_size)
        self.tail = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, out_size),
            nn.Softmax(1)
        )

    def forward(self, batch: bt.IndexedDataBundle):
        features = batch.bundle.features.tensor
        embeddings = self.embedding(torch.tensor(batch.bundle.vocab.vocab_id.values))
        embeddings = embeddings[None, :].repeat(features.shape[0], 1, 1)
        concatenated = torch.cat((features, embeddings), dim=2)

        return self.tail(self.lstm(concatenated))


class PunctNetworkNavec(nn.Module):
    def __init__(self, embedding_size, features_size, hidden_size, out_size, dropout=0):
        super().__init__()
        path_to_navec = Loc.tg_path / 'projects/punct/navec_hudlit_v1_12B_500K_300d_100q.py'
        self.navec = Navec.load(path_to_navec)
        self.NAVEC_EMB_SIZE = 300

        self.embedding = nn.Sequential(
            NavecEmbedding(self.navec),
            nn.Dropout(dropout),
            nn.Linear(self.NAVEC_EMB_SIZE, embedding_size),
            nn.LeakyReLU(0.1)
        )
        self.lstm = btc.LSTMNetwork(embedding_size + features_size, hidden_size)
        self.tail = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, out_size),
            nn.Softmax(1)
        )

    def forward(self, batch: bt.IndexedDataBundle):
        features = batch.bundle.features.tensor
        embeddings = self.embedding(torch.tensor(batch.bundle.navec.navec_id.values))
        embeddings = embeddings[None, :].repeat(features.shape[0], 1, 1)
        concatenated = torch.cat((features, embeddings), dim=2)

        return self.tail(self.lstm(concatenated))


def punct_network_factory_embedding(
        input,
        embedding_size,
        hidden_size,
        out_size,
        dropout=0,
        num_embeddings=10_001
        ):
    features_size = input.bundle.features.tensor.shape[2]

    return PunctNetworkEmbedding(embedding_size, features_size, hidden_size, out_size, dropout, num_embeddings)


def punct_network_factory_navec(
        input,
        embedding_size,
        hidden_size,
        out_size,
        dropout=0
        ):
    features_size = input.bundle.features.tensor.shape[2]

    return PunctNetworkNavec(embedding_size, features_size, hidden_size, out_size, dropout)
