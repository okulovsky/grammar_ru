from grammar_ru.features.architecture import SimpleFeaturizer
from grammar_ru.common import Separator
from unittest import TestCase
import pandas as pd

pd.options.display.width=None


class TestFeaturizer(SimpleFeaturizer):
    def __init__(self, level='word_id'):
        super(TestFeaturizer, self).__init__('features')
        self.level = level
        self.value = 1

    def _featurize_inner(self, db):
        df = db.src[[self.level]].drop_duplicates().set_index(self.level)
        df['value'] =self.value*100+df.index
        return df

