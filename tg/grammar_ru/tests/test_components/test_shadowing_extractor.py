from tg.common.ml import batched_training as bt
from tg.grammar_ru.common import Loc
from unittest import TestCase
from tg.grammar_ru.components import ShadowingTransformer, CoreExtractor
from yo_fluq_ds import *

pd.options.display.width = None

db = bt.DataBundle.load(Loc.test_bundle)
idb = bt.IndexedDataBundle(db.src, db)


case_columns = ['pymorphy_case_gent','pymorphy_case_nomn','pymorphy_case_accs','pymorphy_case_loct','pymorphy_case_ablt','pymorphy_case_datv']

def check(df, prefix):
    case_columns = [c for c in df.columns if c.startswith(prefix)]
    df = db.pymorphy[['POS']].merge(df[case_columns], left_index=True, right_index=True)
    df = df.groupby('POS').sum()
    return df


class ShadowingExtractorTestCase(TestCase):
    def test_without_shadowing(self):
        core = CoreExtractor(allow_list=['pymorphy'])
        core.fit(idb)
        df = core.extract(idb)
        df = check(df, 'pymorphy_case')
        self.assertEqual(0, (df.loc[['NOUN','ADJF'], case_columns]==0).sum().sum())

    def test_with_full_shadowing(self):
        core = CoreExtractor(allow_list=['pymorphy'])
        ex = core.extractors['pymorphy'].extractor
        ex.transformer = ShadowingTransformer(ex.transformer, 'case')
        core.fit(idb)
        df = core.extract(idb)
        df = check(df, 'pymorphy_case')
        self.assertListEqual(['pymorphy_case_NULL'], list(df.columns))

    def test_with_full_exception(self):
        core = CoreExtractor(allow_list=['pymorphy'])
        ex = core.extractors['pymorphy'].extractor
        ex.transformer = ShadowingTransformer(ex.transformer, 'case', 'POS', ['NOUN'])
        core.fit(idb)
        df = core.extract(idb)
        df = check(df, 'pymorphy_case')
        self.assertEqual(6, (df.loc[['ADJF'], case_columns] == 0).sum().sum())
        self.assertEqual(0, (df.loc[['NOUN'], case_columns] == 0).sum().sum())












