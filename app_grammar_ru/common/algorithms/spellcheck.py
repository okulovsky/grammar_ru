import pandas as pd
from .architecture import NlpAlgorithm, DataBundle
import enchant



class SpellcheckAlgorithm(NlpAlgorithm):
    def __init__(self):
        self.spellchecker = enchant.Dict('ru_RU')

    def _run_inner(self, db: DataBundle, index: pd.Index):
        df = db.src.loc[index]
        rdf = pd.DataFrame({}, index=df.index)
        rdf[NlpAlgorithm.Error] = False
        to_check = (df.word_type == 'ru')
        values = df.loc[to_check].word.apply(self.spellchecker.check)
        rdf.loc[to_check, NlpAlgorithm.Error] = ~values

        rdf[NlpAlgorithm.Suggest] = None

        values = df.loc[rdf[NlpAlgorithm.Error]].word.apply(self.spellchecker.suggest)
        rdf.loc[rdf[NlpAlgorithm.Error], NlpAlgorithm.Suggest] = values
        rdf[NlpAlgorithm.ErrorType] = NlpAlgorithm.ErrorTypes.Orthographic
        return rdf

