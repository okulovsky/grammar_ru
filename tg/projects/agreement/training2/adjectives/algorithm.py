from typing import Optional

import pandas as pd

from tg.common import DataBundle
from tg.common.ml.batched_training import IndexedDataBundle
from tg.grammar_ru import Separator
from tg.grammar_ru.algorithms import NlpAlgorithm
from tg.grammar_ru.features import PyMorphyFeaturizer
from tg.projects.agreement.training2.adjectives.endings import extract_ending
from tg.projects.agreement.training2.adjectives.task import AdjectivesTrainingTask
from tg.projects.agreement.training2.declension_extractor import (
    AdjectivesAgreementIndexBuilder,
)


class AdjectivesAgreementAlgorithm(NlpAlgorithm):
    def __init__(self, trained_task: AdjectivesTrainingTask, declension_type: int):
        self._trained_task = trained_task
        self._declension_type = declension_type
        self._index_builder = AdjectivesAgreementIndexBuilder()
        self._featurizer = PyMorphyFeaturizer()

    def get_text_suggestions(self, text: str):
        text_db = Separator.build_bundle(text)
        return self.get_bundle_suggestions(text_db)

    def get_bundle_suggestions(self, db: DataBundle) -> Optional[pd.DataFrame]:
        self._featurizer.featurize(db)

        index_df = self._index_builder.build_index(self._declension_type, db)
        if index_df.empty:
            return None

        input_idb = IndexedDataBundle(index_frame=index_df, bundle=db)
        input_idb.index_frame["label"] = -1
        pred_df = self._trained_task.predict(input_idb)
        return self._condense_predictions(pred_df)

    def _condense_predictions(self, pred_df):
        prefix = "predicted_label_"
        pred_columns = pred_df.columns[pred_df.columns.str.startswith(prefix)]
        pred_labels = pred_df[pred_columns].idxmax(axis=1)
        pred_labels = pred_labels.apply(
            lambda column_name: int(column_name.split("_")[-1])
        )
        condensed_df = pd.DataFrame(pred_labels, columns=["label"])
        condensed_df["ending"] = condensed_df.label.apply(self._get_label_ending)
        return condensed_df

    def _get_label_ending(self, label_index: int) -> str:
        return self._index_builder.get_ending_from_index(
            self._declension_type, label_index
        )

    def _run_inner(self, db: DataBundle, index: pd.Index) -> Optional[pd.DataFrame]:
        db.src = db.src.loc[index]
        suggestions = self.get_bundle_suggestions(db)
        if suggestions is None:
            return None

        suggestions["word"] = db.src.loc[suggestions.index].word
        suggestions["current_ending"] = suggestions.word.apply(extract_ending)
        suggestions = suggestions[suggestions.ending != suggestions.current_ending]

        if suggestions.empty:
            return None

        rdf = pd.DataFrame({}, index=index)
        rdf[NlpAlgorithm.Error] = False

        rdf.loc[suggestions.index, NlpAlgorithm.Error] = True

        rdf[NlpAlgorithm.Suggest] = None
        suggestions["suggestion"] = (
            suggestions[["word", "current_ending", "ending"]]
            .apply(lambda row: self._replace_word_ending(*row), axis=1)
            .values
        )
        rdf.loc[suggestions.index, NlpAlgorithm.Suggest] = suggestions["suggestion"]

        rdf[NlpAlgorithm.ErrorType] = NlpAlgorithm.ErrorTypes.Grammatic
        return rdf

    def _replace_word_ending(
        self, word: str, current_ending: str, new_ending: str
    ) -> str:
        return word[: -len(current_ending)] + new_ending
