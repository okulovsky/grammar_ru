import numpy as np

from tg.projects.agreement.training2.adjectives.endings import (
    extract_ending,
    NEW_num_by_end,
    GOOD_num_by_end,
    BIG_num_by_end,
)


def _replace_end_by_num(df, dt, num_by_end):
    mask = df.declension_type == dt
    df.loc[mask, "label"] = df[mask].ending.map(num_by_end)


# declension_type
# Новый - 0
# Хороший - 1
# Большой - 2


class AdjectivesAgreementIndexBuilder:
    def __init__(self):
        self.norm_endings_nums = {e: i for i, e in enumerate(["ый", "ий", "ой"])}

    def _extract_norm_ending(self, word_in_norm_form: str):
        for possible_ending in self.norm_endings_nums.keys():
            if word_in_norm_form.lower().endswith(possible_ending):
                return possible_ending
        return np.nan

    def build_index(self, declension_type: int, db):
        morphed = db.data_frames["pymorphy"]
        morphed = morphed.replace({np.nan: "nan"})
        adjectives = db.src[(morphed.POS == "ADJF")].copy()

        index_df = db.src[["word_id", "sentence_id"]].copy()
        index_df = index_df.reset_index(drop=True)
        index_df.index.name = "sample_id"

        index_df["declension_type"] = -1

        adjectives["ending"] = adjectives.word.apply(extract_ending)

        morphed_adjectives = morphed.loc[adjectives.index]
        adjectives["norm_ending"] = morphed_adjectives.normal_form.apply(
            self._extract_norm_ending
        )

        undefined_ending_mask = (
            adjectives.norm_ending.isnull() | adjectives.ending.isnull()
        )

        adjectives = adjectives[~undefined_ending_mask]
        adjectives["declension_type"] = adjectives.norm_ending.replace(
            self.norm_endings_nums
        )

        index_df.loc[adjectives.index, "declension_type"] = adjectives[
            "declension_type"
        ].astype(int)
        return index_df[index_df.declension_type == declension_type]

    def get_ending_from_index(self, declension_type: int, index: int) -> str:
        num_by_end = [NEW_num_by_end, GOOD_num_by_end, BIG_num_by_end][declension_type]
        end_by_num = {n: e for e, n in num_by_end.items()}
        return end_by_num[index]
