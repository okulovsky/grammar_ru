import numpy as np

from tg.common import DataBundle
from tg.common.ml.batched_training import train_display_test_split
from tg.grammar_ru.common import Loc
from tg.grammar_ru.corpus import ITransfuseSelector
from tg.projects.agreement.adjectiveless_pymorphy_featurizer import (
    AdjectivelessPyMorphyFeaturizer,
)
from tg.projects.agreement.bundles_tools import _print_thrown

NEW = {
    "ая",
    "ого",
    "ое",
    "ой",
    "ом",
    "ому",
    "ую",
    "ые",
    "ый",
    "ым",
    "ыми",
    "ых"
}
# NOTE выкинули 'ою'

GOOD = {
    "ая",
    "его",
    "ее",
    "ей",
    "ем",
    "ему",
    "ие",
    "ий",
    "им",
    "ими",
    "их",
    "ую",
    "яя",
    "юю",
    "ого",
    "ое",
    "ой",
    "ому",
    "ом",
}  # легкий

BIG = {
    "ая",
    "ие",
    "им",
    "ими",
    "их",
    "ого",
    "ое",
    "ой",
    "ом",
    "ому",
    "ую",
    "ые",
    "ым",
    "ыми",
    "ых",
}  # золотой
# NOTE выкинули 'ою'

NEW_list = sorted(list(NEW))
GOOD_list = sorted(list(GOOD))
BIG_list = sorted(list(BIG))
# окончания с повторами. это фича.
ALL_ENDS_list = NEW_list + GOOD_list + BIG_list
POSSIBLE_ENDINGS = set(ALL_ENDS_list)
endings_nums = {e: i for i, e in enumerate(ALL_ENDS_list)}

NEW_num_by_end = {e: i for i, e in enumerate(NEW_list)}
GOOD_num_by_end = {e: i + len(NEW_num_by_end) for i, e in enumerate(GOOD_list)}
BIG_num_by_end = {
    e: i + len(NEW_num_by_end) + len(GOOD_num_by_end) for i, e in enumerate(BIG_list)
}

nums_by_decl_and_end = (
        {("new", e): n for e, n in NEW_num_by_end.items()}
        | {("good", e): n for e, n in GOOD_num_by_end.items()}
        | {("big", e): n for e, n in BIG_num_by_end.items()}
)


def _extract_ending(word: str):
    for possible_ending in POSSIBLE_ENDINGS:
        if word.lower().endswith(possible_ending):
            return possible_ending
    return np.nan


def _replace_end_by_num(df, dt, num_by_end):
    mask = df.declension_type == dt
    df.loc[mask, "label"] = df[mask].ending.map(num_by_end)
    # df.loc[mask, "label"] = df[mask].ending


# declension_type
# Новый - 0
# Хороший - 1
# Большой - 2


class AdjAgreementIndexBuilder:
    def __init__(self):
        # self.snowball = SnowballStemmer(language="russian")
        self.norm_endings_nums = {e: i for i, e in enumerate(["ый", "ий", "ой"])}
        # self.endings_nums = {e: i for i, e in enumerate(ALL_ENDS_list)}

    def _extract_norm_ending(self, word_in_norm_form: str):
        for possible_ending in self.norm_endings_nums.keys():
            if word_in_norm_form.lower().endswith(possible_ending):
                return possible_ending
        return np.nan

    def build_index(self, db, decl_type):
        # self.pmf.featurize(db)
        morphed = db.data_frames["pymorphy"]
        morphed = morphed.replace({np.nan: "nan"})
        adjectives = db.src[(morphed.POS == "ADJF")].copy()  # TODO delete

        index_df = db.src[["word_id", "sentence_id"]].copy()
        index_df = index_df.reset_index(drop=True)
        index_df.index.name = "sample_id"

        index_df["declension_type"] = -1

        adjectives["ending"] = adjectives.word.apply(_extract_ending)

        morphed_adjectives = morphed.loc[adjectives.index]
        adjectives["norm_ending"] = morphed_adjectives.normal_form.apply(
            self._extract_norm_ending
        )

        undefined_ending_mask = (
                adjectives.norm_ending.isnull() | adjectives.ending.isnull()
        )

        adjectives = adjectives[~undefined_ending_mask]
        print(adjectives.norm_ending)
        adjectives["declension_type"] = adjectives.norm_ending.replace(
            self.norm_endings_nums
        )
        # num_by_end = [NEW_num_by_end, GOOD_num_by_end, BIG_num_by_end][decl_type]
        # _replace_end_by_num(adjectives, decl_type, num_by_end)
        # adjectives = adjectives[~adjectives.label.isnull()]

        index_df.loc[adjectives.index, "declension_type"] = adjectives["declension_type"].astype(int)
        return index_df[index_df.declension_type == decl_type]

    def get_ending_from_index(self, decl_type: int, index: int) -> str:
        num_by_end = [NEW_num_by_end, GOOD_num_by_end, BIG_num_by_end][decl_type]
        end_by_num = {n: e for e, n in num_by_end.items()}
        return end_by_num[index]

# first_declension_ends = set("а я ы и е у ю ой ёй ей ".split())
# # дядей, землёй Note в печатных текстах, наверное, ё заменяют на е
# # second_declension_ends = set("а я ы и е у ю ой ёй ей ою ".split())
# POSSIBLE_ENDINGS = first_declension_ends

# ends_list = sorted(list(first_declension_ends))
# num_by_end = {e: n for n, e in enumerate(ends_list)}


# def _extract_ending(word: str):
#     for possible_ending in POSSIBLE_ENDINGS:
#         if word.lower().endswith(possible_ending):
#             return possible_ending
#     return np.nan


class NounAgreementTrainIndexBuilder(ITransfuseSelector):
    first_declension_ends = set("а ы е у ой".split())
    # дядей, землёй Note в печатных текстах, наверное, ё заменяют на е

    POSSIBLE_ENDINGS = first_declension_ends
    ends_list = sorted(list(first_declension_ends))
    num_by_end = {e: n for n, e in enumerate(ends_list)}
    end_by_num = {n: e for (e, n) in num_by_end.items()}

    def _extract_ending(self, word: str):
        for possible_ending in self.POSSIBLE_ENDINGS:
            if word.lower().endswith(possible_ending):
                return possible_ending
        return np.nan

    def __init__(self):
        self.pmf = AdjectivelessPyMorphyFeaturizer()
        # self.snowball = SnowballStemmer(language="russian")
        self.norm_endings_nums = {e: i for i, e in enumerate(["а"])}
        # self.endings_nums = {e: i for i, e in enumerate(ALL_ENDS_list)}

    def _extract_norm_ending(self, word_in_norm_form: str):
        for possible_ending in self.norm_endings_nums.keys():
            if word_in_norm_form.lower().endswith(possible_ending):
                return possible_ending
        return np.nan

    def select(self, source, df, toc_row):
        db = DataBundle(src=df)
        self.pmf.featurize(db)
        morphed = db.data_frames["pymorphy"]
        morphed.replace({np.nan: "nan"}, inplace=True)
        nouns = df[(morphed.POS == "NOUN")].copy()  # TODO delete
        # return morphed[(morphed.POS == "NOUN")]
        df["is_target"] = False
        df["declension_type"] = -1

        nouns["ending"] = nouns.word.apply(self._extract_ending)

        morphed_nouns = morphed.loc[nouns.index]
        nouns["norm_ending"] = morphed_nouns.normal_form.apply(
            self._extract_norm_ending
        )

        undefined_ending_mask = nouns.norm_ending.isnull() | nouns.ending.isnull()
        thrown = list(set(nouns[undefined_ending_mask].word))

        nouns = nouns[~undefined_ending_mask]
        nouns["declension_type"] = 1
        # adjectives.norm_ending.replace(            self.norm_endings_nums)

        nouns["label"] = nouns.ending.map(self.num_by_end)
        thrown.extend(nouns[nouns.label.isnull()].word)
        nouns = nouns[~nouns.label.isnull()]

        df.loc[nouns.index, "declension_type"] = nouns["declension_type"]
        df.loc[nouns.index, "norm_ending"] = nouns["norm_ending"].map(
            self.norm_endings_nums
        )
        df.declension_type = df.declension_type.astype(int)
        df["label"] = -1
        df.loc[nouns.index, "label"] = nouns.label
        df.loc[nouns.index, "is_target"] = True
        _print_thrown(thrown, Loc.temp_path / "noun_undefined_ending.txt")
        return [df]

    @staticmethod
    def build_index_from_src(src_df):
        df = src_df.loc[src_df.is_target][
            ["word_id", "sentence_id", "declension_type", "label"]
        ].copy()
        df = df.reset_index(drop=True)
        df.index.name = "sample_id"
        df["split"] = train_display_test_split(df)
        return df
