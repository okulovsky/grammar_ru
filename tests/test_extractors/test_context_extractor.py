from grammar_ru.analyzers.natasha.natasha_syntax_analyzer import NatashaSyntaxAnalyzer
from grammar_ru.common.architecture.separator import Separator
from grammar_ru.extractors.slovnet_context_extractor import ContextExtractor
from grammar_ru.common.natasha import create_chunks_from_dataframe
from unittest import TestCase
import pandas as pd
from tg.common.ml import batched_training as bt

text = 'Она была красива. Он любил красивые вещи. Вещи, нитрокраситель и нитроэмаль!'


class ContextExtractorTestCase(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super(ContextExtractorTestCase, cls).setUpClass()
        cls.syntax = NatashaSyntaxAnalyzer()
        cls.df = Separator.separate_string(text)
        chunks = create_chunks_from_dataframe(cls.df)
        cls.syntax_df = cls.syntax.analyze_chunks(cls.df, chunks)
        print(cls.syntax_df)
        index_df = pd.DataFrame(cls.df["word_id"], columns=["word_id"]).set_index("word_id")
        cls.bundle = bt.DataBundle(index_df, dict(
            syntax=cls.syntax_df
        ))
        cls.extractor = ContextExtractor("Syntax")
        cls.result = cls.extractor.extract(index_df, cls.bundle)
        print(cls.result)

    def test_syntax_general(self):
        self.assertEqual(self.result[(self.result["word_id"] == 0) & (self.result["relative_word_id"] == 0)]["shift"].item(), 0)
        self.assertEqual(self.result[(self.result["word_id"] == 0) & (self.result["relative_word_id"] == 2)]["shift"].item(), 1)
        self.assertEqual(self.result[(self.result["word_id"] == 1) & (self.result["relative_word_id"] == 2)]["shift"].item(), 1)
        try:
            self.result[(self.result["word_id"] == 2) & (self.result["shift"] == 1)]["relative_word_id"].item()
            self.fail()
        except BaseException:
            pass

