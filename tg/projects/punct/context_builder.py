from tg.common.ml import batched_training as bt
from tg.grammar_ru.components import PlainContextBuilder
import pandas as pd


class PunctContextBuilder(PlainContextBuilder):
    def build_context(self, ibundle: bt.IndexedDataBundle, context_size: int) -> pd.DataFrame:
        context_df = super().build_context(ibundle, context_size)
        context_df = context_df.reset_index()
        word_is_target = (
            ibundle.bundle.src
            .set_index('word_id')
            .loc[context_df.sample_id].is_target
            .values
        )
        context_df = context_df[~(word_is_target & (context_df.offset == 1))]

        return context_df.set_index(['sample_id', 'offset'])
