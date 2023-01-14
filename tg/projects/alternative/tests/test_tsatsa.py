from unittest import TestCase
from tg.grammar_ru.ml.tasks.tsatsa import TsaTsaTask, TaskBuilder
from tg.grammar_ru.common import Separator
from tg.grammar_ru.ml.features import PyMorphyFeaturizer, SlovnetFeaturizer

class TsaTsaTestCase(TestCase):
    def test_transform(self):
        task = TsaTsaTask()
        task.good_words_ = {'нравится','нравиться'}
        db = Separator.build_bundle(
            'Нравится нравиться понравится понравиться'
        )

        dfs = task.process(db)
        self.assertListEqual(['Нравиться', 'нравится', 'понравится', 'понравиться'], list(dfs[1].word))
        self.assertListEqual([1, 1, 1, 1], list(dfs[1].label))
        self.assertListEqual([1, 1, 0, 0], list(dfs[1].is_target))

        self.assertListEqual(['Нравится', 'нравиться', 'понравится', 'понравиться'], list(dfs[0].word))
        self.assertListEqual([0,0, 0, 0], list(dfs[0].label))
        self.assertListEqual([1, 1, 0, 0], list(dfs[0].is_target))

    def test_preview(self):
        task = TsaTsaTask()
        db = Separator.build_bundle(
            'Нравится нравиться смотреться уебаться уебатся'
        )
        task.preview([db])
        self.assertSetEqual({'нравиться', 'нравится'}, task.good_words_)

    def test_full_cycle(self):
        task = TsaTsaTask()
        db = Separator.build_bundle(
            'Хуй поклав. Мне нравится всем нравиться, но не нравится уебываться.'
        )
        builder = TaskBuilder(None, task)
        result = task.build(None, lambda: [db], [PyMorphyFeaturizer(), SlovnetFeaturizer()])
        self.assertSetEqual({'src', 'pymorphy', 'slovnet'}, set(result.data_frames))
        self.assertListEqual(
            ['Мне', 'нравится', 'всем', 'нравиться', ',', 'но', 'не', 'нравится', 'уебываться', '.', 'Мне', 'нравиться', 'всем', 'нравится', ',', 'но', 'не', 'нравиться', 'уебываться', '.'],
            list(result.src.word)
        )
        self.assertListEqual(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            list(result.src.word_id)
        )
        self.assertListEqual([0, 19],
                             list(result.src.sentence_id.unique()))







