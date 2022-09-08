from ....common.ml.batched_training import context as btc
from ....common.ml import batched_training as bt
import pandas as pd
import numpy as np


class PlainContextBuilder(btc.ContextBuilder):
    def __init__(self,
                 include_zero_offset = False,
                 left_to_right_contexts_proportion: float = 0):
        rows = []
        for i in range(1, 50):
            for j in range(1, i + 1):
                rows.append(dict(context_length=i, offset=j))
        self.of_df = pd.DataFrame(rows).set_index('context_length')
        self.include_zero_offset = include_zero_offset
        self.left_to_right_contexts_proportion = left_to_right_contexts_proportion

    def build_partial_context(self,
                                index_frame: pd.DataFrame,
                                sentence_limits: pd.DataFrame,
                                context_size: int,
                                is_left: bool
                           ):
        df = index_frame.merge(sentence_limits, left_on='sentence_id', right_index=True)

        if is_left:
            df['cl'] = df.word_id - context_size
            df.cl = np.maximum(df.cl, df.sentence_begin)
            df.cl = df.word_id - df.cl
        else:
            df['cl'] = df.word_id + context_size
            df.cl = np.minimum(df.cl, df.sentence_end)
            df.cl = df.cl - df.word_id

        df = df[['word_id', 'cl']].merge(self.of_df, left_on='cl', right_index=True)

        if is_left:
            df.offset *= -1

        df['another_word_id'] = df.word_id + df['offset']
        df = df.set_index('offset', append=True)[['another_word_id']]
        return df

    def get_left_and_right_sizes(self, context_size):
        if self.include_zero_offset:
            context_size-=1
        if self.left_to_right_contexts_proportion<=0:
            return context_size, 0
        if self.left_to_right_contexts_proportion>=1:
            return 0, context_size
        right = int(round((1-self.left_to_right_contexts_proportion)*context_size))
        return context_size - right, right


    def build_context(self, ibundle: bt.IndexedDataBundle, context_size) -> pd.DataFrame:
        sl_df = (ibundle.bundle.src.loc[ibundle.bundle.src.sentence_id.isin(ibundle.index_frame.sentence_id)]
                .groupby('sentence_id')
                .word_id.aggregate(['min', 'max'])
                .rename(columns=dict(min='sentence_begin', max='sentence_end'))
                 )
        frames = []
        left_size, right_size = self.get_left_and_right_sizes(context_size)
        if self.include_zero_offset:
            frames.append(ibundle.index_frame[['word_id']]
                          .rename(columns=dict(word_id='another_word_id'))
                          .assign(offset=0)
                          .set_index('offset',append=True)
                          )
        if left_size>0:
            frames.append(self.build_partial_context(ibundle.index_frame, sl_df, left_size, True))
        if right_size>0:
            frames.append(self.build_partial_context(ibundle.index_frame, sl_df, right_size, False))
        return pd.concat(frames)



