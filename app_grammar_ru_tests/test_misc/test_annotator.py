from unittest import TestCase
from tg.grammar_ru.misc.annotator import DummyTaskProvider, Annotator

class AnnotatorTestCase(TestCase):
    def test_dummy_task(self):
        prov = DummyTaskProvider()
        task_1 = prov.get()
        prov.store(None)
        task_2 = prov.get()
        prov.undo()
        task_3 = prov.get()
        print([task_1.__dict__, task_2.__dict__, task_3.__dict__])