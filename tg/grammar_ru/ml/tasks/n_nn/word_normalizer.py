import re
import abc

from nltk.stem import SnowballStemmer

from .regular_expressions import single_n_regex, double_n_regex


class WordNormalizer(abc.ABC):
    @abc.abstractmethod
    def normalize_word(self, word: str) -> str:
        pass


class NltkWordStemmer(WordNormalizer):
    def __init__(self) -> None:
        self._stemmer = SnowballStemmer(language='russian')

    def normalize_word(self, word: str) -> str:
        return self._stemmer.stem(word)


class RegexNormalizer(WordNormalizer):
    def normalize_word(self, word: str) -> str:
        if (match := re.search(single_n_regex, word)):
            return word[:match.start() + 1]
        elif (match := re.search(double_n_regex, word)):
            return word[:match.start()]

        return word


class EmptyNormalizer(WordNormalizer):
    """For debug purposes"""

    def normalize_word(self, word: str) -> str:
        return word
