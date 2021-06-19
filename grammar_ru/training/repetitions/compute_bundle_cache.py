from grammar_ru.algorithms import RepetitionsAlgorithm
from research.amenities import *
from copy import deepcopy


class BundleCreator:
    def __init__(self, enrichers):
        self.bundles = []
        self.alg = RepetitionsAlgorithm(100, True, True, True, True)
        self.enrichers = enrichers

    def build_index(self, db):
        idf = db.data_frames['src']
        idf['check_requested'] = True
        self.alg.run(idf)
        self.last_.after_algorithm = idf

        idf = idf.loc[~idf.repetition_status]
        idf = idf[['word_id', 'file_id', 'repetition_algorithm', 'repetition_reference']]
        idf.columns = ['word_id', 'file_id', 'match_type', 'another_word_id']

        idf = idf.merge(db.data_frames['pymorphy'][['POS']], left_on='word_id', right_index=True)
        idf = idf.loc[~idf.POS.isin(['NPRO', 'PREP', 'CONJ', 'PRCL'])]
        idf = idf.drop('POS', axis=1)
        self.last_.after_filtration = idf

        return idf



    def __call__(self, db):
        self.last_ = Obj()
        self.last_.incoming = deepcopy(db)
        for enricher in self.enrichers:
            enricher(db)
        idf = self.build_index(db)
        db.index_frame = idf
        wids = list(idf.word_id) + list(idf.another_word_id)
        filter_bundle_by_words(db, wids)
        return db

enrichers = [
    add_capitalization_data,
    add_local_freq,
    WordFrequencyFeaturizer()
]

opath = Loc.bundles_path / 'repetitions/cache'
cr = BundleCreator(enrichers)



def filt(toc):
    toc['file_id'] = toc.index
    exists = toc.file_id.apply(lambda z: (opath / z).is_dir())
    return (toc.token_count > 10000) & ~exists

if __name__ == '__main__':
    (Corpus
     .get_bundles(filt)
     .feed(fluq.with_progress_bar())
     .select(cr)
     .foreach(lambda z: z.save(opath / z.index_frame['file_id'].iloc[0]))
     )