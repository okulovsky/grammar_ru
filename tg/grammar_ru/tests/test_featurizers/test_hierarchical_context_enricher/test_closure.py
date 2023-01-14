from tg.grammar_ru.ml.features.hierarchical_context_featurizer import build_closure
from unittest import TestCase
from tg.grammar_ru.common import Separator
from tg.grammar_ru.ml.features import SlovnetFeaturizer
from tg.grammar_ru.common.tree_viewer import TreeViewer

class ClosureTestCase(TestCase):
    def test_simple(self):
        db = Separator.build_bundle(
            'Несмотря на сильный ветер, он двигался вперед',
            [SlovnetFeaturizer()]
        )
        self.assertListEqual(['case', 'fixed', 'amod', 'obl', 'punct', 'nsubj', 'root', 'advmod'], list(db.slovnet.relation))
        self.assertListEqual([3, 0, 3, 6, 3, 6, -1, 6], list(db.slovnet.syntax_parent_id))

        viewer = TreeViewer(0, db.src).add_relation(db.slovnet.syntax_parent_id, db.slovnet.relation).add_labels(db.src.word+' '+db.src.word_id.astype(str)).draw()
        reldf = db.slovnet
        cl = build_closure(reldf, 10)

        self.assertListEqual([0, 1, 2, 3, 4, 5, 7, 0, 2, 4, 1, 1], list(cl.word_id))
        self.assertListEqual([3, 0, 3, 6, 3, 6, 6, 6, 6, 6, 3, 6], list(cl.syntax_parent_id))
        self.assertListEqual([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3], list(cl.distance))
        self.assertListEqual(['case', 'fixed', 'amod', 'obl', 'punct', 'nsubj', 'advmod', 'obl', 'obl', 'obl', 'case', 'obl'], list(cl.relation))
