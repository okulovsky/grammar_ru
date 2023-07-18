from .architecture import *
import snowballstemmer

class SnowballFeaturizer(SimpleFeaturizer):
    def __init__(self, language = 'russian'):
        super(SnowballFeaturizer, self).__init__('snowball')
        self.stemmer = snowballstemmer.stemmer(language)
    
    def _featurize_inner(self, db: DataBundle):
        words = db.src.word.str.lower().str.replace('ё', 'e')
        stems = self.stemmer.stemWords(words)
        endings = [word[len(stem):] for word, stem in zip(words, stems)]
        return pd.DataFrame(dict(
            word = words,
            stem = stems,
            ending = endings
        ))
        