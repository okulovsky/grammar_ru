import pandas as pd
import numpy as np

class EnglishLocalizator:
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

            if word == '\u201c':
                self.dialog_sentence = True
            if word == '\u201d':
                self.dialog_sentence_closed = True

        return cur_sentence
    