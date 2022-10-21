from tg.grammar_ru.ml.features.architecture import *
import pandas as pd
import numpy as np
from yo_fluq_ds import fluq

DIALOG_DASH = ['—', '–', '-']

PRE_DASH = [
    '.', ',', '?', '!', '…', ':', '?!',
    '...', '?..', '!..', '!!!', '!!', '..', '….',
    '!?', ';', '.?', '.!', '…?'
]

NO_DIALOG_PUNCTUATION = ['»', '"', ')', '–', '“', '”', '(', '«']

def _get_dialog_paragraphs(df):
    df = df.feed(fluq.add_ordering_column('paragraph_id','sentence_id','sentence_index'))
    paragraphs = df.loc[(df.word_index==0) & (df.sentence_index==0) & (df.word.isin(DIALOG_DASH))].paragraph_id
    tdf = df.loc[df.paragraph_id.isin(paragraphs)].copy()
    tdf['next_word'] = tdf.word.shift(-1).fillna('')
    tdf['next_word_type'] = tdf.word_type.shift(-1).fillna('')
    tdf['next_paragraph_id'] = tdf.paragraph_id.shift(-1).fillna(-1).astype(int)
    return tdf


def _get_borders(tdf):
    kdf = tdf.loc[
        (
            (tdf.word.isin(PRE_DASH)) &
            (tdf.next_word_type=='punct') &
            (~tdf.next_word.isin(NO_DIALOG_PUNCTUATION)) &
            (tdf.paragraph_id==tdf.next_paragraph_id)
        )
        | (tdf.sentence_index == 0)
    ].copy()
    kdf['border_word_id'] = kdf.word_id
    kdf['border_word_id_2'] = np.where(kdf.word_index==0,-1,kdf.word_id+1)
    kdf = kdf.feed(fluq.add_ordering_column('paragraph_id','word_id','border_index'))
    return kdf


def _build_markup(tdf, kdf):
    sdf = tdf[['word_id','paragraph_id', 'word']].merge(
        kdf.set_index('paragraph_id')[['border_word_id','border_word_id_2','border_index']],
        left_on='paragraph_id',
        right_index=True
    )
    sdf = sdf.loc[sdf.word_id>=sdf.border_word_id]
    sdf = sdf.feed(fluq.add_ordering_column('word_id',('border_word_id',False), 'border_order'))
    sdf = sdf.loc[sdf.border_order==0]
    sdf['dialog_type'] = np.where(
        ((sdf.word_id==sdf.border_word_id) | (sdf.word_id==sdf.border_word_id_2)),
        np.where(
            sdf.word.isin(DIALOG_DASH),
            'dialog-dash',
            'dialog-symbol'
        ),
        np.where(
            sdf.border_index%2==0,
            'speech',
            'action'
        )
    )
    sdf['is_dialog'] = True
    sdf = sdf.set_index('word_id')[['dialog_type', 'is_dialog']]
    return sdf


def _finalize_markup(df, sdf):
    udf = df.set_index('word_id')[[]].merge(sdf, left_index=True, right_index=True, how='left')
    udf.dialog_type=udf.dialog_type.fillna('no')
    udf.is_dialog = udf.is_dialog.fillna(False)
    return udf


class DialogMarkupFeaturizer(SimpleFeaturizer):
    def __init__(self):
        super(DialogMarkupFeaturizer, self).__init__('dialog_markup', False)

    def _featurize_inner(self, db: DataBundle):
        df = db.src
        tdf = _get_dialog_paragraphs(df)
        kdf = _get_borders(tdf)
        sdf = _build_markup(tdf, kdf)
        udf = _finalize_markup(df, sdf)
        return udf








