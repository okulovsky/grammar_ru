from typing import *
from ...common import Separator, Loc
from .corpus_writer import CorpusWriter
import pandas as pd
from pathlib import Path
from ....common._common import Logger

class CorpusBufferedWriter:
    def __init__(self,
                 path: Path,
                 words_per_frame: int,
                 break_down_by_sentence = False
                 ):
        self.path = path
        self.words_per_frame = words_per_frame
        self.break_down_by_sentence = break_down_by_sentence

    def open(self):
        self.writer_ = CorpusWriter(self.path, True, None)
        self.buffer_ = []
        self.delta_ = 0

    def _write_simple(self):
        sdf = pd.concat(self.buffer_)
        Logger.info(f'Writing {len(self.buffer_)} frames, {sdf.shape[0]} words')
        self.writer_.add_fragment(sdf)
        self.buffer_ = []

    def _write_with_breakdown(self, finalize = False):
        while True:
            sdf = pd.concat(self.buffer_, sort = False)
            sids = sdf.groupby('sentence_id').size().cumsum()
            sids = sids.loc[sids<self.words_per_frame].index
            to_write = sdf.loc[sdf.sentence_id.isin(sids)]
            to_keep = sdf.loc[~sdf.sentence_id.isin(sids)]
            Logger.info(f'Writing {len(self.buffer_)} frames, {sdf.shape[0]} words, to write {to_write.shape[0]}, to keep {to_keep.shape[0]}')
            self.writer_.add_fragment(to_write)
            self.buffer_ = [to_keep]
            if not finalize:
                if self._words_in_buffer() < self.words_per_frame:
                    break
            if finalize:
                if self._words_in_buffer() == 0:
                    break



    def _flush(self, finalize = False):
        if len(self.buffer_)==0:
            return
        if not self.break_down_by_sentence:
            self._write_simple()
            return
        else:
            self._write_with_breakdown(finalize)
            return

    def _words_in_buffer(self) -> int:
        return sum(x.shape[0] for x in self.buffer_)

    def add(self, df):
        if not hasattr(self,'writer_'):
            raise ValueError('`CorpusBufferedWriter is not opened')
        if df.shape[0]==0:
            return
        df = Separator.reset_indices(df, self.delta_)
        self.delta_ = Separator.get_max_id(df) + 10
        Separator.validate(df)
        self.buffer_.append(df)
        sum_length = sum([x.shape[0] for x in self.buffer_])
        if sum_length>self.words_per_frame:
            self._flush()


    def close(self):
        self._flush(True)
        self.writer_.finalize()