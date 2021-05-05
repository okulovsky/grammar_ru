from ..preprocessors.dummy_preprocessor import DummyPreprocessor
from ..common import NlpAlgorithm
import pandas as pd
import enchant


class SpellcheckAlgorithm(NlpAlgorithm):
    def __init__(self):
        super(SpellcheckAlgorithm, self).__init__(DummyPreprocessor(), 'spellcheck_status', 'spellcheck_suggestion')
        self.spellchecker = enchant.Dict('ru_RU')

    def _run_inner(self, df: pd.DataFrame):
        column = self.get_status_column()
        df[column] = True
        to_check = (df.word_type == 'ru') & (df.check_requested)
        values = df.loc[to_check].word.apply(self.spellchecker.check)
        df.loc[to_check, column] = values

        suggest_column = self.get_suggest_column()
        df[suggest_column] = None
        to_suggest = ~df[column]
        values = df.loc[to_suggest].word.apply(self.spellchecker.suggest)
        df.loc[to_suggest, suggest_column] = values
