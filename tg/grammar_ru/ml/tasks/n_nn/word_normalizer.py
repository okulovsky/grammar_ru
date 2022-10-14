import abc

from nltk.stem import SnowballStemmer


class WordNormalizer(abc.ABC):
    @abc.abstractmethod
    def normalize_word(self, word: str) -> str:
        pass


class NltkWordStemmer(WordNormalizer):
    def __init__(self) -> None:
        self._stemmer = SnowballStemmer(language='russian')

    def normalize_word(self, word: str) -> str:
        return self._stemmer.stem(word)
