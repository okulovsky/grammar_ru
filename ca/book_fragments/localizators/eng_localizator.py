import pandas as pd

class EnglishLocalizator:
    def __init__(self) -> None:
        self.dialog_sentence = False
        self.dialog_sentence_closed = False

    def construct_sentence(self, frame: pd.DataFrame, sentence_id: int):
        cur_sentence = []

        for word_id in frame[frame['sentence_id'] == sentence_id]['word_id'].unique().tolist():
            word = frame[frame['word_id'] == word_id]['word'].values[0]

            if word == '\u00A0':
                word = '\u0020'

            cur_sentence.append(
                word + ' ' * frame[frame['word_id']
                                   == word_id]['word_tail'].values[0]
            )

            if frame[frame['word_id'] == word_id]['word'].values[0] == '\u201c':
                self.dialog_sentence = True
            if frame[frame['word_id'] == word_id]['word'].values[0] == '\u201d':
                self.dialog_sentence_closed = True

        return cur_sentence
    