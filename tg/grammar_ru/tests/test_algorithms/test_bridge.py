from unittest import TestCase
from tg.grammar_ru.algorithms import AlgorithmBridge, SpellcheckAlgorithm
from tg.common.delivery.packaging import make_package, PackagingTask, install_package_and_get_loader
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.width = None

class BridgeTestCase(TestCase):
    def test_bridge(self):
        bridge = AlgorithmBridge(SpellcheckAlgorithm(), debug = True)
        data = bridge.run(['Окно','Акно'])
        self.assertEqual(0, len(data.dumps))
        self.assertEqual([False,True], list(data.result_df.error))

        data = bridge.run(['X', 'Икно', 'X'], [0, None, 1], data)
        self.assertEqual(0, len(data.dumps))
        self.assertListEqual([False,True,False], list(data.result_df.updated))
        self.assertListEqual([False,True,False], list(data.result_df.updated))

    def test_bridge_on_empty(self):
        bridge = AlgorithmBridge(SpellcheckAlgorithm(), debug=True)
        data = bridge.run([])
        self.assertIsNone(data.result_df)
        self.assertIsNone(data.bundle)

    def test_bridge_with_package(self):
        source_bridge = AlgorithmBridge(SpellcheckAlgorithm(), debug=True)
        pkg = make_package(PackagingTask('grammar_ru', '', dict(entry=source_bridge)))
        bridge = install_package_and_get_loader(pkg.path).load_resource('entry')
        data = bridge.run(['Окно', 'Акно'])
        self.assertEqual([False,True], list(data.result_df.error))




