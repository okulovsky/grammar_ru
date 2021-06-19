from yo_fluq_ds import *
from .corpus_reader import CorpusReader

class CorpusCombined:
    def __init__(self, corpuses: List[CorpusReader]):
        self.corpuses = corpuses

    def _get_corps_and_tocs(self, filter, fragments_to_take):
        tocs = [z.get_toc() for z in self.corpuses]
        if filter is not None:
            tocs = [t.loc[filter(t)] for t in tocs]
        if fragments_to_take is not None:
            tocs = [t.iloc[:fragments_to_take] for t in tocs]
        length = sum(z.shape[0] for z in tocs)
        return tocs, length

    def get_frames(self, filter = None, fragments_to_take=None):
        tocs, length = self._get_corps_and_tocs(filter, fragments_to_take)
        return Queryable(
            Query.en(zip(self.corpuses,tocs)).select_many(lambda z: z[0].get_frames(z[1].index)),
            length
        )

    def get_tocs(self):
        tocs, _ = self._get_corps_and_tocs(None, None)
        return pd.concat(tocs)

    def get_bundles(self, filter = None, fragments_to_take=None):
        tocs, length = self._get_corps_and_tocs(filter, fragments_to_take)
        return Queryable(
            Query.en(zip(self.corpuses, tocs)).select_many(lambda z: z[0].get_bundles(z[1].index)),
            length
        )
