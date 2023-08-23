import nltk
import razdel
import re
from typing import *

from .abstract_separator import AbstractSeparator
from .symbols import Symbols


class SeparatorRuClass(AbstractSeparator):
    def __init__(self):
        super().__init__()
        self._russianRegex = re.compile('^[{0}]+$'.format(re.escape(Symbols.RUSSIAN_WORD_SYMBOLS)))
        self._punctuation = re.compile('^[{0}]+$'.format(re.escape(Symbols.PUNCTUATION)))

    def _tokenize(self, text: str) -> Iterable[str]:
        return [token.text for token in razdel.tokenize(text)]

    def _sentenize(self, text: str) -> Iterable[str]:
        return [sent.text for sent in razdel.sentenize(text)]

    def _classify_word(self, word: str) -> str:
        return 'punct' if self._punctuation.match(word) else ('ru' if self._russianRegex.match(word) else 'unk')

    def from_word_en(self, words: Iterable[str]):
        return AbstractSeparator._from_word_en(words, 'ru')


class SeparatorEnClass(AbstractSeparator):

    def __init__(self):
        self._en_Regex = re.compile('^[{0}]+$'.format(re.escape(Symbols.EN_WORD_SYMBOLS)))
        self._punctuation = re.compile('^[{0}]+$'.format(re.escape(Symbols.EN_PUNCT)))

    def _tokenize(self, text: str) -> Iterable[str]:
        return nltk.word_tokenize(text)

    def _sentenize(self, text: str) -> Iterable[str]:
        return nltk.sent_tokenize(text)

    def _classify_word(self, word: str) -> str:
        return 'punct' if self._punctuation.match(word) else ('en' if self._en_Regex.match(word) else 'unk')

    def _to_str(self, token) -> str:
        return token

    def from_word_en(self, words: Iterable[str]):
        return AbstractSeparator._from_word_en(words, 'en')


Separator = SeparatorRuClass()
SeparatorRu = SeparatorRuClass()
SeparatorEn = SeparatorEnClass()
