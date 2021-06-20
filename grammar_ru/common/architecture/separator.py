from typing import *
from razdel import sentenize, tokenize
import pandas as pd
import re


def _generate_offsets(string: str, substrings: List[str]):
    offset = 0
    result = []
    for s in substrings:
        while not string.startswith(s, offset):
            offset += 1
            if offset >= len(string):
                raise ValueError(f'Substring failed for {string}, {s}')
        result.append(offset)
        offset += len(s)
    return result, offset


class Symbols:
    RUSSIAN_LETTERS = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    RUSSIAN_WORD_SYMBOLS = RUSSIAN_LETTERS+'-'+chr(8242)
    PUNCTUATION = ',.–?!—…:«»";()“”„-'
    APOSTROPHS = "'"+chr(8217)




class Separator:
    russianRegex = re.compile('^[{0}]+$'.format(re.escape(Symbols.RUSSIAN_WORD_SYMBOLS)))
    punctuation = re.compile('^[{0}]+$'.format(re.escape(Symbols.PUNCTUATION)))

    @staticmethod
    def _classify_word(word):
        return 'punct' if Separator.punctuation.match(word) else ('ru' if Separator.russianRegex.match(word) else 'unk')

    @staticmethod
    def _separate_string(s: str, word_id_start=0, sentence_id_start=0):
        result = []
        sentences = sentenize(s)
        offset = 0
        word_id = word_id_start
        sentence_id = sentence_id_start
        for sentence in sentences:
            tokens = [token.text for token in tokenize(sentence.text)]
            offsets, delta = _generate_offsets(s[offset:], tokens)
            for word_index, (word, word_offset) in enumerate(zip(tokens, offsets)):
                word_type = Separator._classify_word(word)
                result.append((word_id, sentence_id, word_index, offset+word_offset, word, word_type))
                word_id += 1
            offset += delta
            sentence_id += 1
        df = pd.DataFrame(result, columns=['word_id', 'sentence_id', 'word_index', 'word_offset', 'word', 'word_type'])
        df['word_length'] = df.word.str.len()
        return df

    @staticmethod
    def separate_string(s: str):
        return Separator.separate_paragraphs(s.split('\n'))

    @staticmethod
    def separate_paragraphs(strings: List[str]):
        result = []
        word_id_start, sentence_id_start = 0, 0
        for i, s in enumerate(strings):
            df = Separator._separate_string(s, word_id_start, sentence_id_start)
            if df.shape[0] > 0:
                word_id_start = df.iloc[-1].word_id+1
                sentence_id_start = df.iloc[-1].sentence_id+1
            df['paragraph_id'] = i
            result.append(df)
        return pd.concat(result, ignore_index=True)
