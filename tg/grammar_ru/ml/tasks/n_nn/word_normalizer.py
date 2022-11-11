import typing as tp
from collections import defaultdict
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
            return word[:match.start() + 1].lower()
        elif (match := re.search(double_n_regex, word)):
            return word[:match.start()].lower()

        return word.lower()


class LoggingRegexNormalizer(RegexNormalizer):
    """For debug purposes"""
    def __init__(self) -> None:
        self._map: tp.MutableMapping[str, tp.List[str]] = defaultdict(list)

    def normalize_word(self, word: str) -> str:
        normalized = super().normalize_word(word)
        self._map[normalized].append(word)

        return normalized
