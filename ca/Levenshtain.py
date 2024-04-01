import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Any
from grammar_ru.common.separator.df_viewer import DfViewer
from torch.nn.functional import cosine_similarity
from grammar_ru import SeparatorEn, SeparatorRu, Separator
from tg.projects.retell.retell_utils.Embeders import AbstractEmbedder
from tqdm.auto import tqdm


class LevenshtainAlg:
    def __init__(self, lang, embeder: Optional[AbstractEmbedder] = None, threshold=0.7):
        self.sep: Separator
        if lang == 'ru':
            self.sep = SeparatorRu
        elif lang == 'en':
            self.sep = SeparatorEn
        else:
            raise ValueError("Unknown language. It must be 'ru' or 'en'")
        self.highlighter: DfViewer = self.sep.Viewer().highlight('Lev_status', {"Matched": 'green', "Deleted": "red",
                                                                                "Transited": "yellow",
                                                                                "MatchedWithTransited": "yellow"})
        self.b_del_penalty = 1
        self.rt_del_penalty = 3
        self.substitution_penalty = 2
        self.substitution_penalty_calc = lambda cos_sim: max(0, 10 * (1 - cos_sim) - 1)
        self.embeder = embeder
        self.threshold = threshold
        self.ans_seq: List[str] = []

    def do_alg(self, book_chapter_df: pd.DataFrame, retell_chapter_df: pd.DataFrame) -> Tuple[
        int, pd.DataFrame, pd.DataFrame]:
        n, m = len(book_chapter_df) + 1, len(retell_chapter_df) + 1
        all_book_tokens, all_rtl_tokens = book_chapter_df.word.str.lower().values, retell_chapter_df.word.str.lower().values
        # Однозначное соотвествие,  ch_word = chapter_df.word.values[i-1],rtl_word = retell_chapter_df.word.values[j-1]
        # Если при восстановлении ответа некоторые некоторые варианты равны, то сначала берём удаление из главы, потом замену, потом удаление из пересказа.
        self.state_matrix = np.zeros((n, m), dtype='float16')
        self.state_matrix[0] = np.arange(m)
        self.state_matrix[..., 0] = np.arange(n)
        for i in tqdm(range(1, n)):
            for j in range(1, m):
                book_token = all_book_tokens[i - 1]
                retell_token = all_rtl_tokens[j - 1]
                delete_from_book = self.state_matrix[i - 1][j] + self.b_del_penalty  # удаление из главы самая дешёвая
                delete_from_retell = self.state_matrix[i][
                                         j - 1] + self.rt_del_penalty  # удаление из пересказа самая дорогая
                substitution = self.state_matrix[i - 1][j - 1] + self.m_func(book_token,
                                                                             retell_token)  # замена слова на слово, в том числе если они равны
                best_choice = min(delete_from_book, delete_from_retell, substitution)
                self.state_matrix[i][j] = best_choice

        book_chapter_df, retell_chapter_df = self.restore_ans_seq(book_chapter_df, retell_chapter_df)

        ans_alg = self.state_matrix[-1][-1]
        return ans_alg, book_chapter_df, retell_chapter_df

    def restore_ans_seq(self, book_chapter_df: pd.DataFrame, retell_chapter_df: pd.DataFrame):
        book_chapter_df['Lev_status'] = [False] * len(book_chapter_df)
        retell_chapter_df['Lev_status'] = [False] * len(retell_chapter_df)
        book_chapter_df['Transited_to_id'] = [None] * len(book_chapter_df)
        book_chapter_df['Transited_to_word'] = [None] * len(book_chapter_df)
        book_chapter_df['Matched_with'] = [None] * len(book_chapter_df)
        book_chapter_df['Penalty'] = [0] * len(book_chapter_df)

        n, m = len(book_chapter_df), len(retell_chapter_df)
        all_book_word_id, all_rtl_word_id = book_chapter_df.word_id.values, retell_chapter_df.word_id.values
        all_book_tokens, all_rtl_tokens = book_chapter_df.word.values, retell_chapter_df.word.values
        i, j = n, m
        while i > 0 and j > 0:
            book_word_id, rtl_word_id = all_book_word_id[i - 1], all_rtl_word_id[j - 1]
            delete_from_book = self.state_matrix[i - 1][j]
            delete_from_retell = self.state_matrix[i][j - 1]
            substitution = self.state_matrix[i - 1][j - 1]

            best_choice = min(delete_from_book, delete_from_retell, substitution)

            (book_lev_status, rtl_lev_status, book_penalty, rtl_penalty,
             book_matched_with, rtl_matched_with, transited_to_id,
             transited_to_word) = '', '', 0, 0, None, None, None, None

            if best_choice == delete_from_book:
                book_lev_status, book_penalty = 'Deleted', self.b_del_penalty
                i -= 1
            elif best_choice == delete_from_retell:
                rtl_lev_status, rtl_penalty = 'Deleted', self.rt_del_penalty
                j -= 1
            else:
                book_token, retell_token = all_book_tokens[i - 1], all_rtl_tokens[j - 1]
                if book_token.lower() == retell_token.lower():
                    book_lev_status, book_matched_with = 'Matched', rtl_word_id
                    rtl_lev_status, rtl_matched_with = 'Matched', book_word_id
                else:
                    cos_sim = self.__get_cos_sim_from_tokens(book_token, retell_token)
                    book_lev_status, transited_to_id, transited_to_word, book_penalty = (
                        'Transited', rtl_word_id, retell_token,
                        self.substitution_penalty_calc(cos_sim))
                    rtl_lev_status, rtl_matched_with = 'MatchedWithTransited', book_word_id
                i -= 1
                j -= 1

            book_chapter_df = self.__set_values(book_chapter_df, book_word_id,
                                                ['Lev_status', 'Matched_with', 'Transited_to_id', 'Transited_to_word',
                                                 'Penalty'],
                                                [book_lev_status, book_matched_with, transited_to_id, transited_to_word,
                                                 book_penalty])

            retell_chapter_df = self.__set_values(retell_chapter_df, rtl_word_id,
                                                  ['Lev_status', 'Matched_with', 'Penalty'],
                                                  [rtl_lev_status, rtl_matched_with, rtl_penalty])
        if i > 0:
            for q in range(i):
                book_word_id = all_book_word_id[q - 1]
                self.__set_values(book_chapter_df, book_word_id,
                                  ['Lev_status', 'Penalty'],
                                  ["Deleted", self.b_del_penalty])
        if j > 0:
            for q in range(j):
                rtl_word_id = all_rtl_word_id[q - 1]
                self.__set_values(retell_chapter_df, rtl_word_id,
                                  ['Lev_status', 'Penalty'],
                                  ["Deleted", self.rt_del_penalty])

        return book_chapter_df, retell_chapter_df

    def highlight(self, df) -> Any:
        return self.highlighter.to_html_display(df)

    def __get_cos_sim_from_tokens(self, f_token, s_token):
        chp_emb, rtl_emb = [self.embeder.get_embedding(token) for token in [f_token, s_token]]
        return cosine_similarity(chp_emb, rtl_emb, dim=0).item()

    def __set_values(self, df, id, columns, values):
        for column, value in zip(columns, values):
            df.at[id, column] = value
        return df

    def m_func(self, chp_token, rtl_token):
        if chp_token == rtl_token:
            self.ans_seq.append(chp_token)
            return 0
        cos_sim = self.__get_cos_sim_from_tokens(chp_token, rtl_token)
        if cos_sim > self.threshold:
            self.ans_seq.append(rtl_token)
            return self.substitution_penalty_calc(cos_sim)  # max = [0,2], 2 = 10 * (1 - 0.7) - 1
        else:
            return self.substitution_penalty
