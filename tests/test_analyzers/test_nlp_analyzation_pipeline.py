from grammar_ru.analyzers.natasha.natasha_morph_analyzer import NatashaMorphAnalyzer
from grammar_ru.analyzers.natasha.natasha_syntax_analyzer import NatashaSyntaxAnalyzer
from grammar_ru.analyzers.natasha.combined_natasha_analyzer import CombinedNatashaAnalyzer
from grammar_ru.common.architecture.nlp_analyzation_pipeline import NlpAnalyzationPipeline
from grammar_ru.common.architecture.separator import Separator
from grammar_ru.common.architecture.validations import ensure_df_contains
from unittest import TestCase
import numpy as np

text = 'Она была красива. Он любил красивые вещи. Вещи, нитрокраситель и нитроэмаль!'


class NlpAnalyzationPipelineTestCase(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super(NlpAnalyzationPipelineTestCase, cls).setUpClass()
        cls.analyzer = NlpAnalyzationPipeline(
            [("SlovnetSyntaxMorph", CombinedNatashaAnalyzer([NatashaMorphAnalyzer(), NatashaSyntaxAnalyzer()]))])

    def test_morph_and_syntax_general(self):
        # TODO: Unit tests
        print(self.analyzer.analyze_string(text))
