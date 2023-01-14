from tg.grammar_ru.features.architecture import SimpleFeaturizer
from tg.grammar_ru.common import Separator
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



class UpdateTestCase(TestCase):
    def check(self,  ft: TestFeaturizer):
        old_bundle = Separator.build_bundle('Первый абзац\nВторой абзац\nТретий абзац', [ft])
        ft.value += 1
        new_bundle = Separator.update_bundle(
            old_bundle,
            ['Первый новый', 'Третий абзац', 'Второй новый', 'Первый абзац', 'Третий новый'],
            [None, 2, None, 0, None],
            [ft]
        )
        return old_bundle, new_bundle

    def test_update(self):
        old_bundle, new_bundle = self.check(TestFeaturizer())
        self.assertListEqual(
            [200, 201, 104, 105, 204, 205, 100, 101, 208, 209],
            list(new_bundle.features.sort_index().value)
        )


    def test_update_on_sentence_level(self):
        old_bundle, new_bundle = self.check(TestFeaturizer(level='sentence_id'))
        self.assertListEqual(
            [200, 102, 202, 100, 204],
            list(new_bundle.features.sort_index().value)
        )
