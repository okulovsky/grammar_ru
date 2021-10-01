from grammar_ru.ml.corpus import CorpusCombined, CorpusReader
from pathlib import Path

class LocHolder:
    def __init__(self):
        self.data_path = (Path(__file__).parent/'../../../data').absolute()
        self.corpus_path = self.data_path/'corpus'
        self.bundles_path = self.data_path/'bundles'

Loc = LocHolder()


Corpus = CorpusCombined([
    CorpusReader(Loc.corpus_path/'books.zip'),
    CorpusReader(Loc.corpus_path/'proza.zip', 100000000)
])

