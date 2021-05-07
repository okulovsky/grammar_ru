import pymorphy2
from yo_fluq_ds import *
from .featurizer import Featurizer

class PyMorphyFeaturizer(Featurizer):

    def create_features(self, df):
        morph = pymorphy2.MorphAnalyzer()
        pymorphy_tags = ['POS', 'animacy', 'aspect', 'case', 'gender', 'involvement', 'mood', 'number', 'person', 'tense',
                         'transitivity', 'voice']
        rrows = []
        for row in Query.df(df):
            index = row.word_id
            result = morph.parse(row.word)[0]
            rrow = [index, result.normal_form]
            for t in pymorphy_tags:
                rrow.append(getattr(result.tag, t))
            rrows.append(rrow)
        pdf = pd.DataFrame(rrows, columns=['word_id','normal_form'] + pymorphy_tags)
        return pdf

