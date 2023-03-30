from tg.common.ml import batched_training as bt
from tg.grammar_ru.components import PlainContextBuilder
import pandas as pd


class PunctContextBuilder(PlainContextBuilder):
    def build_context(self, ibundle: bt.IndexedDataBundle, context_size) -> pd.DataFrame:
        context_df = super().build_context(ibundle, context_size)
        context_df = context_df.reset_index()
        word_is_target = (
            ibundle.bundle.src
            .set_index('word_id')
            .loc[context_df.sample_id].is_target
            .values
        )
        max_offset = context_df.offset.max()
        rows_to_drop = (~(word_is_target & (context_df.offset == 1))) & ~(~word_is_target & (context_df.offset == max_offset))
        context_df = context_df[rows_to_drop]

        return context_df.set_index(['sample_id', 'offset'])
