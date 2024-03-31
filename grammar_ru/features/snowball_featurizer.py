from .architecture import *
from nltk.stem.snowball import RussianStemmer, EnglishStemmer
from nltk import pos_tag


# import nltk

# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')


class SnowballFeaturizer(SimpleFeaturizer):
    def __init__(self, language='rus'):
        super(SnowballFeaturizer, self).__init__('snowball')
        self.language = language
        self.stemmer = RussianStemmer() if language == 'rus' else EnglishStemmer()

    def _featurize_inner(self, db: DataBundle):
        index = db.src.word_id
        words = db.src['word'].str.lower()
        if self.language == 'rus':
            words = words.str.replace('ё', 'e')
        words = words.values
        stems = [self.stemmer.stem(word) for word in words]
        pos = [tag[1] for tag in pos_tag(tokens=words, lang=self.language, tagset='universal')]
        endings = [word[len(stem):] for word, stem in zip(words, stems)]
        return pd.DataFrame(dict(
            word=words,
            stem=stems,
            ending=endings,
            pos=pos
        ), index=index)
