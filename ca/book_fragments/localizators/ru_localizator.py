import pandas as pd
import numpy as np

class RussianLocalizator:
    def __init__(self) -> None:
        self.dialog_sentence = False
        self.dialog_sentence_closed = False

    def construct_sentence(self, frame: pd.DataFrame, sentence_id: int):
        cur_sentence = []

        for word_id in frame.iloc[np.where(frame.sentence_id.isin([sentence_id]))]['word_id'].unique():
            word = frame.iloc[np.where(frame.word_id.isin([word_id]))]['word'].values[0]

            if word == '\u00A0':
                word = '\u0020'

            cur_sentence.append(
                word + ' ' * frame[frame['word_id']
                                   == word_id]['word_tail'].values[0]
            )

            if word == '\u2014':
                self.dialog_sentence = True

        paragraph_id = frame.iloc[np.where(frame.sentence_id.isin([sentence_id]))]['paragraph_id'].unique()[0] 
        last_paragraph_sentence_id = frame.iloc[np.where(frame.paragraph_id.isin([paragraph_id]))]['sentence_id'].unique()[0] 

        if last_paragraph_sentence_id == sentence_id:
            self.dialog_sentence_closed = True

        return cur_sentence