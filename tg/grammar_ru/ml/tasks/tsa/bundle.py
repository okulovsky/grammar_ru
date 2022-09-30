from typing import *
from ....common import DataBundle
import pandas as pd
from ....algorithms import SpellcheckAlgorithm
import numpy as np
from .....common.ml.batched_training import train_display_test_split

def _count_verbs(dfs: Iterable[pd.DataFrame]):
    counter = {}
    for df in dfs:
        df = df.loc[df.word.str.endswith('тся') | df.word.str.endswith('ться')]
        for word in df.word.str.lower():
            counter[word] = counter.get(word,0)+1
    return counter


def _another(w):
    if w.endswith('тся'):
        return w.replace('тся', 'ться')
    else:
        return w.replace('ться', 'тся')


def _get_good_words_df(words):
    for w in list(words):
        if _another(w) not in words:
            words[_another(w)] = 0

    rows = []
    for w in words:
        if w.endswith('ться'):
            rows.append((w,_another(w),words[w], words[_another(w)]))

    df = pd.DataFrame(rows, columns=['i_word','f_word','i_cnt','f_cnt'])
    df['both_found'] = df[['i_cnt','f_cnt']].min(axis=1)>0
    df = df[df.both_found]
    df['ratio'] = np.minimum(df.i_cnt/df.f_cnt, df.f_cnt/df.i_cnt)

    alg = SpellcheckAlgorithm()
    for prefix in ['i_', 'f_']:
        xdf = pd.DataFrame(dict(word=df[prefix+'word'], word_type='ru'))
        kdf = alg.run(DataBundle(src=xdf))
        df[prefix+'spell'] = ~kdf.error

    df['both_correct'] = df.i_spell & df.f_spell
    df = df.loc[df.both_correct]
    return df


def build_dictionary(dfs):
    cnt = _count_verbs(dfs)
    df = _get_good_words_df(cnt)
    good_words = set(df.i_word).union(df.f_word)
    return good_words


class TrainIndexBuilder:
    def __init__(self, good_words, add_negative_samples = True):
        self.good_words = good_words
        self.ref_id = 0
        self.add_negative_samples = add_negative_samples

    def build_train_index(self, df):
        ddf = df.iloc[[0]]
        description = (ddf.corpus_id+'/'+ddf.file_id).iloc[0]

        df['original_corpus_id'] = df.corpus_id
        df['is_target'] = df.word.str.lower().isin(self.good_words)
        good_sentences = df.groupby('sentence_id').is_target.max().feed(lambda z: z.loc[z].index)
        positive = df.loc[df.sentence_id.isin(good_sentences)].copy()
        positive['label'] = 0

        ref_map = {v: self.ref_id + k for k, v in enumerate(positive.sentence_id.unique())}
        positive['reference_sentence_id'] = positive.sentence_id.replace(ref_map)
        self.ref_id += 1 + len(positive.sentence_id.unique())

        ar = [positive]

        if self.add_negative_samples:
            negative = positive.copy()
            negative.word = np.where(
                ~negative.is_target,
                negative.word,
                np.where(
                    negative.word.str.endswith('тся'),
                    negative.word.str.replace('тся', 'ться'),
                    negative.word.str.replace('ться', 'тся')
                )
            )
            negative['label'] = 1
            ar = [positive, negative]

        for f in ar:
            if f.sentence_id.isnull().any():
                raise ValueError(f"Null sentence id when processing, uid {description}")
        return ar

    @staticmethod
    def build_index_from_src(src_df):
        df = src_df.loc[src_df.is_target][['word_id', 'sentence_id', 'label', 'reference_sentence_id']].copy()
        df = df.reset_index(drop=True)
        df.index.name = 'sample_id'
        df['split'] = train_display_test_split(df)
        return df
