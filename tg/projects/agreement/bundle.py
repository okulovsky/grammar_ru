import numpy as np
import pandas as pd
from tg.grammar_ru.common import Loc

import re

from tg.common import DataBundle
from tg.common.ml.batched_training import train_display_test_split
from tg.grammar_ru.features import PyMorphyFeaturizer

from tg.grammar_ru.corpus import ITransfuseSelector
from nltk.stem import SnowballStemmer
from pymystem3 import Mystem

mystem = Mystem()

new = {'ая', 'ого', 'ое', 'ой', 'ом', 'ому',
       'ою', 'ую', 'ые', 'ый', 'ым', 'ыми', 'ых'}

good = {'ая', 'его', 'ее', 'ей', 'ем', 'ему',
        'ие', 'ий', 'им', 'ими', 'их', 'ую', 'яя', 'юю'}

big = {'ая', 'ие', 'им', 'ими', 'их', 'ого',
       'ое', 'ой', 'ом', 'ому', 'ою', 'ую'}

POSSIBLE_ENDINGS = set().union(new, good, big)


def _get_poses_by_sentence(sentence: str):
    # NOTE: краткие прилагательные mystem'ом отмечаются как прилагательные. e.g. Хороша, плох.
    res = []
    for word_info in mystem.analyze(sentence):
        if 'analysis' not in word_info or not word_info["analysis"]:
            continue
        res.append(
            (word_info["text"],
             re.split(
                 ',|=', word_info["analysis"][0]["gr"])[0])
        )
    return res


def _set_mystem_pos(df):
    df['pos_mystem'] = np.nan
    for sent_id, tokens_group in df.groupby("sentence_id"):
        sentence = ' '.join(tokens_group.word)
        poses = _get_poses_by_sentence(sentence)
        if not poses:
            continue
        j = 0
        for i in tokens_group.index:  # zip with gaps
            word, pos = poses[j]
            if df.at[i, 'word'] == word:
                df.at[i, 'pos_mystem'] = pos
                j += 1
                if j == len(poses):
                    break


def _extract_true_ending(stemmed_ending: str):
    for possible_ending in POSSIBLE_ENDINGS:  # TODO can we make it faster?
        if stemmed_ending.lower().endswith(possible_ending):
            return possible_ending
    return np.nan


class AdjAgreementTrainIndexBuilder(ITransfuseSelector):
    def __init__(self):
        self.pmf = PyMorphyFeaturizer()
        self.snowball = SnowballStemmer(language="russian")
        self.norm_endings_nums = {e: i for i, e in enumerate(['ый', 'ий', 'ой'])}
        self.endings_nums = {e: i for i, e in enumerate(
            sorted(list(POSSIBLE_ENDINGS)))}

    def select(self, source, df, toc_row):  # ~build_train_index
        _set_mystem_pos(df)
        db = DataBundle(src=df)
        self.pmf.featurize(db)  # запишет результат по ключу pymorphy
        morphed = db.data_frames['pymorphy']
        morphed.replace({np.nan: 'nan'}, inplace=True)
        adjectives = df[
            (df.pos_mystem == 'A') &
            (morphed.POS == "ADJF")
            ].copy()  # TODO delete
        df['is_target'] = False
        df['declension_type'] = -1

        adjectives['ending'] = (adjectives.word
                                .apply(_extract_true_ending))

        morphed_adjectives = morphed.loc[adjectives.index]
        adjectives['norm_ending'] = (morphed_adjectives.normal_form
                                     .apply(_extract_true_ending))

        # adjectives['norm_form'] = morphed_adjectives.normal_form

        with open(Loc.temp_path / "undefined_ending.txt", "a") as myfile:
            for w in adjectives[
                adjectives.norm_ending.isnull() |
                adjectives.ending.isnull()
            ].word:
                myfile.write(f'{w}\n')
        adjectives = adjectives[
            ~adjectives.norm_ending.isnull() &
            ~adjectives.ending.isnull()
            # NOTE: отбросили слова, у которых не смогли определить окончание. e.g. волчий
            ]
        df.loc[adjectives.index, 'declension_type'] = adjectives.norm_ending.replace(
            self.norm_endings_nums)
        df['label'] = -1
        df.loc[adjectives.index, 'label'] = adjectives.ending.replace(
            self.endings_nums)
        df.loc[adjectives.index, 'is_target'] = True
        return [df]

    @staticmethod
    def build_index_from_src(src_df):
        df = src_df.loc[src_df.is_target][[
            'word_id', 'sentence_id', 'label']].copy()
        df = df.reset_index(drop=True)
        df.index.name = 'sample_id'
        df['split'] = train_display_test_split(df)
        return df
