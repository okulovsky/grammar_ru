import numpy as np
import pandas as pd
from tg.grammar_ru.common import Loc
from collections import defaultdict

import re

from tg.common import DataBundle
from tg.common.ml.batched_training import train_display_test_split
from tg.grammar_ru.features import PyMorphyFeaturizer

from tg.grammar_ru.corpus import ITransfuseSelector
from nltk.stem import SnowballStemmer


def _print_thrown(thrown):
    with open(Loc.temp_path / "undefined_ending.txt", "a") as myfile:
        for w in thrown:
            myfile.write(f'{w}\n')


NEW = {'ая', 'ого', 'ое', 'ой', 'ом', 'ому',
       'ую', 'ые', 'ый', 'ым', 'ыми', 'ых'}
# NOTE выкинули 'ою'

GOOD = {'ая', 'его', 'ее', 'ей', 'ем', 'ему',
        'ие', 'ий', 'им', 'ими', 'их', 'ую', 'яя', 'юю',
        'ого','ое', 'ой', 'ому', 'ом'} # легкий

BIG = {'ая', 'ие', 'им', 'ими', 'их', 'ого',
       'ое', 'ой', 'ом', 'ому', 'ую',
       'ые', 'ым', 'ыми', 'ых'} # золотой
# NOTE выкинули 'ою'

NEW_list = sorted(list(NEW))
GOOD_list = sorted(list(GOOD))
BIG_list = sorted(list(BIG))
# окончания с повторами. это фича.
ALL_ENDS_list = NEW_list + GOOD_list + BIG_list
POSSIBLE_ENDINGS = set(ALL_ENDS_list)
endings_nums = {e: i for i, e in enumerate(ALL_ENDS_list)}

NEW_num_by_end = {e: i for i, e in enumerate(NEW_list)}
GOOD_num_by_end = {e: i+len(NEW_num_by_end) for i, e in enumerate(GOOD_list)}
BIG_num_by_end = {e: i+len(NEW_num_by_end)+len(GOOD_num_by_end)
                  for i, e in enumerate(BIG_list)}

nums_by_decl_and_end = (
    {('new', e): n for e, n in NEW_num_by_end.items()} |
    {('good', e): n for e, n in GOOD_num_by_end.items()} |
    {('big', e): n for e, n in BIG_num_by_end.items()}
)


def _extract_ending(word: str):
    for possible_ending in POSSIBLE_ENDINGS:
        if word.lower().endswith(possible_ending):
            return possible_ending
    return np.nan

# declension_type
# Новый - 0
# Хороший - 1
# Большой - 2


class AdjAgreementTrainIndexBuilder(ITransfuseSelector):
    def __init__(self):
        self.pmf = PyMorphyFeaturizer()
        # self.snowball = SnowballStemmer(language="russian")
        self.norm_endings_nums = {e: i for i,
                                  e in enumerate(['ый', 'ий', 'ой'])}
        # self.endings_nums = {e: i for i, e in enumerate(ALL_ENDS_list)}

    def _extract_norm_ending(self, word_in_norm_form: str):
        for possible_ending in self.norm_endings_nums.keys():
            if word_in_norm_form.lower().endswith(possible_ending):
                return possible_ending
        return np.nan

    def select(self, source, df, toc_row):
        db = DataBundle(src=df)
        self.pmf.featurize(db)
        morphed = db.data_frames['pymorphy']
        morphed.replace({np.nan: 'nan'}, inplace=True)
        adjectives = df[(morphed.POS == "ADJF")].copy()  # TODO delete
        df['is_target'] = False
        df['declension_type'] = -1

        adjectives['ending'] = (adjectives.word
                                .apply(_extract_ending))

        morphed_adjectives = morphed.loc[adjectives.index]
        adjectives['norm_ending'] = (morphed_adjectives.normal_form
                                     .apply(self._extract_norm_ending))

        undefined_ending_mask = (adjectives.norm_ending.isnull() |
                                 adjectives.ending.isnull())
        thrown = list(set(adjectives[undefined_ending_mask].word))

        adjectives = adjectives[~undefined_ending_mask]
        adjectives['declension_type'] = adjectives.norm_ending.replace(
            self.norm_endings_nums)
        adjectives.loc[adjectives[adjectives.declension_type == 0].index, 'label'] = adjectives.ending.map(
            NEW_num_by_end)
        adjectives.loc[adjectives[adjectives.declension_type == 1].index, 'label'] = adjectives.ending.map(
            GOOD_num_by_end)
        adjectives.loc[adjectives[adjectives.declension_type == 2].index, 'label'] = adjectives.ending.map(
            BIG_num_by_end)
        thrown.extend(adjectives[adjectives.label.isnull()].word)
        adjectives = adjectives[~adjectives.label.isnull()]

        df.loc[adjectives.index, 'declension_type'] = adjectives['declension_type']
        df.declension_type = df.declension_type.astype(int)
        df['label'] = -1
        df.loc[adjectives.index, 'label'] = adjectives.label
        df.loc[adjectives.index, 'is_target'] = True
        _print_thrown(thrown)
        return [df]

    @staticmethod
    def build_index_from_src(src_df):
        df = src_df.loc[src_df.is_target][[
            'word_id', 'sentence_id', 'declension_type', 'label']].copy()
        df = df.reset_index(drop=True)
        df.index.name = 'sample_id'
        df['split'] = train_display_test_split(df)
        return df
