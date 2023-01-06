import typing as tp

from .train_index_builder import IndexBuilder
from ..n_nn.word_normalizer import WordNormalizer
from tg.grammar_ru.ml.tasks.train_index_builder.sentence_filterer import (
    DictionaryFilterer, ChtobyFilterer, NormalizeFilterer
)
from tg.grammar_ru.ml.tasks.train_index_builder.negative_sampler import (
    ChtobyNegativeSampler, TsaNegativeSampler, NNnNegativeSampler
)


class TsaIndexBuilder(IndexBuilder):
    def __init__(
            self,
            good_words: tp.Sequence[str],
            add_negative_samples: bool = True
            ) -> None:
        filterer = DictionaryFilterer(good_words)
        negative_sampler = TsaNegativeSampler()
        super().__init__(
            filterer=filterer,
            negative_sampler=negative_sampler,
            add_negative_samples=add_negative_samples
        )


class NNnIndexBuilder(IndexBuilder):
    def __init__(
            self,
            good_words: tp.Sequence[str],
            word_normalizer: WordNormalizer,
            add_negative_samples: bool = True):
        filterer = NormalizeFilterer(good_words, word_normalizer)
        negative_sampler = NNnNegativeSampler()
        super().__init__(
            filterer=filterer,
            negative_sampler=negative_sampler,
            add_negative_samples=add_negative_samples)

        self._word_normalizer = word_normalizer


class ChtobyIndexBuilder(IndexBuilder):
    def __init__(self, add_negative_samples: bool = True):
        filterer = ChtobyFilterer()
        negative_sampler = ChtobyNegativeSampler()
        super().__init__(
            filterer=filterer,
            negative_sampler=negative_sampler,
            add_negative_samples=add_negative_samples
        )
