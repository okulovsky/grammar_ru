import spacy
from torch import tensor, Tensor
from abc import ABC, abstractmethod
from .embeddings import get_vocab_embedding


class AbstractEmbedder(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_embedding(self, token: str) -> Tensor:
        raise NotImplementedError("")


class Embeder(AbstractEmbedder):
    def __init__(self, vectorizer, vocab, vec_shape=300):
        super().__init__()
        self.vectorizer = vectorizer
        self.vocab = vocab
        self.vec_shape = vec_shape

    def get_embedding(self, token: str) -> Tensor:
        return get_vocab_embedding(token=token, vectorizer=self.vectorizer, vocab=self.vocab, vec_shape=self.vec_shape)


class SpacyEmbeder(AbstractEmbedder):

    def __init__(self):
        super().__init__()
        self.embeder = spacy.load("en_core_web_md")

    def get_embedding(self, token: str) -> Tensor:
        return tensor(self.embeder(token).vector)
