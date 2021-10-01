from typing import *
from .data_bundle import DataBundle
from razdel import sentenize, tokenize
import pandas as pd
import re
from yo_fluq_ds import Query


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

    COLUMNS = ['word_id', 'sentence_id', 'word_index', 'paragraph_id', 'word_offset', 'word', 'word_type', 'word_length']

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
        rdf = pd.concat(result, ignore_index=True)
        rdf = rdf[Separator.COLUMNS]
        return rdf

    @staticmethod
    def separate_string_into_bundle(s: str):
        return DataBundle(src=Separator.separate_string(s))

    @staticmethod
    def separate_paragraphs_into_bundle(strings: List[str]):
        return DataBundle(src=Separator.separate_paragraphs(strings))

    @staticmethod
    def check_df(df):
        for c in Separator.COLUMNS:
            if c not in df.columns:
                raise ValueError(
                    f"Column {c} is not in dataframe. Use a Separator output as an argument to avoid such problems.")


    @staticmethod
    def _normalize(df):
        df = df.sort_values(['paragraph_id', 'word_id']).copy()
        df.word_id = list(range(df.shape[0]))
        df.index = list(df.word_id)
        sentence_map = Query.en(df.sentence_id.unique()).with_indices().to_dictionary(lambda z: z.value,
                                                                                      lambda z: z.key)
        df = df.replace(dict(sentence_id=sentence_map))
        return df

    @staticmethod
    def update_separation(old_df: pd.DataFrame, new_paragraphs, paragraph_to_id):
        old_map = Query.en(paragraph_to_id).with_indices().where(lambda z: z.value is not None).to_dictionary(
            lambda z: z.value, lambda z: z.key)
        old_df = old_df.loc[old_df.paragraph_id.isin(old_map)].copy()
        old_df['updated'] = False
        old_df['original_paragraph_id'] = old_df.paragraph_id
        old_df['original_sentence_id'] = old_df.sentence_id
        old_df['original_word_id'] = old_df.word_id
        old_df = old_df.replace(dict(paragraph_id=old_map))

        new_map = Query.en(paragraph_to_id).with_indices().where(
            lambda z: z.value is None).with_indices().to_dictionary(lambda z: z.key, lambda z: z.value.key)
        if len(new_map) == 0:
            return Separator._normalize(old_df)

        pars = []
        for i in sorted(new_map.values()):
            pars.append(new_paragraphs[i])
        new_df = Separator.separate_paragraphs(pars)
        new_df = new_df.replace(dict(paragraph_id=new_map))
        new_df.word_id += old_df.word_id.max() + 1000
        new_df.sentence_id += old_df.sentence_id.max() + 1000
        new_df['updated'] = True
        new_df['original_paragraph_id'] = -1
        new_df['original_sentence_id'] = -1
        new_df['original_word_id'] = -1

        df = pd.concat([old_df, new_df], axis=0)
        return Separator._normalize(df)