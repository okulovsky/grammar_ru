from grammar_ru.features.architecture import *
import pandas as pd
import numpy as np


def dialog_trigger(df, buffer, id, dialog_id, in_dialog):
    # TODO optimize pandas
    for word_id, dialog_token_type in buffer:
        df.loc[word_id, ['dialog_type', 'dialog_id', 'dialog_token_type']] = 'dialog', dialog_id, dialog_token_type,
    df.loc[id, 'dialog_token_type'] = 'dialog-dash'
    df.loc[id, 'dialog_type'] = 'dialog'
    dialog_id += 1
    df.loc[id, 'dialog_id'] = dialog_id
    dialog_id += 1
    in_dialog = not in_dialog
    return dialog_id, in_dialog, []


class DialogMarkupFeaturizer(SimpleFeaturizer):
    def __init__(self, dialog_punc):
        super(DialogMarkupFeaturizer, self).__init__('dialog_markup', False)
        self.dialog_punc = dialog_punc

    def _featurize_inner(self, db: DataBundle):
        df = db.src.copy()
        dialog_dash = self.dialog_punc
        dialog_columns = ['dialog_type', 'dialog_id', 'dialog_token_type']
        df[dialog_columns] = 'text', 0, 'none'
        in_dialog = False
        dialog_id = 0
        prev_paragraph_id,prev_paragraph_dialog_id = df.paragraph_id.min(), 0
        dialog_buffer = []

        for row in df.itertuples():
            word, word_type, word_id, sentence_id, paragraph_id = row.word, row.word_type, row.word_id, row.sentence_id, row.paragraph_id

            if prev_paragraph_id != paragraph_id:
                if in_dialog:
                    in_dialog = False
                    df.loc[df.paragraph_id == prev_paragraph_id, dialog_columns] = 'wrong',prev_paragraph_dialog_id + 1, 'none'
                    dialog_id = prev_paragraph_dialog_id + 2

                prev_paragraph_dialog_id = dialog_id
                prev_paragraph_id = paragraph_id

            if in_dialog:
                if word in dialog_dash:  #exit from dialog
                    dialog_id, in_dialog, dialog_buffer = dialog_trigger(df, dialog_buffer, row.Index, dialog_id, in_dialog)
                    #fill dialog with action
                    # TODO optimize pandas
                    df.loc[(df.sentence_id == sentence_id) & (df.word_id < word_id) & (df.dialog_token_type == 'none'), dialog_columns] = 'dialog', dialog_id - 4, 'action'
                    df.loc[(df.sentence_id == sentence_id) & (df.word_id > word_id) & (df.dialog_token_type == 'none'), dialog_columns] = 'dialog', dialog_id, 'action'
                else:
                    dialog_token_type = 'dialog-symbol' if word_type == 'punct' else 'speech'
                    dialog_buffer.append((word_id, dialog_token_type))
            else:
                if word in dialog_dash:
                    dialog_id, in_dialog, dialog_buffer = dialog_trigger(df, [], row.Index, dialog_id, in_dialog)
                else:
                    df.loc[row.Index, 'dialog_id'] = dialog_id
        return df[['word_id', 'dialog_type', 'dialog_id', 'dialog_token_type']]
