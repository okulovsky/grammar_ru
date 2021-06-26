from unittest import TestCase
from grammar_ru.common import DataBundle, Loc
from yo_fluq_ds import *
from pathlib import Path

class DataBundleTestCase(TestCase):
    def test_access(self):
        bundle = DataBundle(index='a', src='b')
        self.assertEqual('a', bundle.data_frames['index'])
        self.assertEqual('a', bundle['index'])
        self.assertEqual('a', bundle.index_frame)
        self.assertEqual('a', bundle.index)

        self.assertEqual('b', bundle.data_frames['src'])
        self.assertEqual('b', bundle['src'])
        self.assertEqual('b', bundle.src)

        bundle['test'] = 'c'
        self.assertEqual('c', bundle.test)
        self.assertEqual('c', bundle.data_frames['test'])

        bundle.index_frame = 'e'
        self.assertEqual('e', bundle.data_frames['index'])

    def test_io(self):
        df1 = Query.en(range(4)).select(lambda z: dict(a=z, b=2*z)).to_dataframe()
        df2 = df1*3
        b = DataBundle(a=df1, b=df2)
        path = Loc.temp_path/'tests/bundle_test/'
        b.save(path)
        b1 = DataBundle.load(path)
        self.assertListEqual([0,1,2,3], list(b1.a.a))
        self.assertListEqual([0, 3, 6, 9], list(b1.b.a))
