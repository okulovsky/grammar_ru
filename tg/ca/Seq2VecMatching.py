from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from tg.grammar_ru import SeparatorRu
from collections import defaultdict


class Seq2VecMatcher:
    def __init__(self, model_name='distiluse-base-multilingual-cased', separator=SeparatorRu):
        self.model = SentenceTransformer(model_name)
        self.viewer = separator.Viewer()
        self._current_group_flag = 1
        self._next_group_flag = 2

    def get_matches(self, df_1: pd.DataFrame, df_2: pd.DataFrame, need_matching_dict=False):
        text_1, text_2 = [self.viewer.to_sentences_strings(df).values for df in [df_1, df_2]]
        encoded_text_1, encoded_text_2 = [self.model.encode(text, convert_to_numpy=True) for text in [text_1, text_2]]
        cos_sim = util.cos_sim(encoded_text_1, encoded_text_2)
        action_matrix = self._get_action_matrix(cos_sim)
        monotone_matching = self._get_monotone_matching(action_matrix)
        matched_sent = {text_1[t_1_id]: text_2[t_2_id] for t_1_id, t_2_id in monotone_matching.items()}
        return matched_sent if not need_matching_dict else (monotone_matching, matched_sent)

    def get_df_matching(self, df_1: pd.DataFrame, df_2: pd.DataFrame):
        text_1, text_2 = [self.viewer.to_sentences_strings(df).values for df in [df_1, df_2]]
        encoded_text_1, encoded_text_2 = [self.model.encode(text, convert_to_numpy=True) for text in [text_1, text_2]]
        cos_sim = util.cos_sim(encoded_text_1, encoded_text_2)
        action_matrix = self._get_action_matrix(cos_sim)
        monotone_matching = self._get_monotone_matching(action_matrix)
        df_1['MatchedWith'] = [0] * len(df_1)
        df_2['MatchedWith'] = [0] * len(df_2)

    def _get_action_matrix(self, cos_sim):
        n, m = cos_sim.shape[0] + 1, cos_sim.shape[1] + 1
        if n < m:
            n, m = m, n
            cos_sim = cos_sim.T
        matrix = np.zeros((n, m))
        actions_matrix = np.zeros((n, m))
        for j in range(2, m):
            actions_matrix[j][j] = self._next_group_flag
        for j in range(1, m):
            matrix[j][j] = cos_sim[j - 1][j - 1].item() + matrix[j - 1][j - 1]

        for i in range(1, n):  # text
            for j in range(1, min(i, m)):  # retell
                sim = cos_sim[i - 1][j - 1].item()  # -1 в индексе из-за того, что матрица ответа обрамлена нулями
                add_to_current_sentences_group = matrix[i - 1][j]
                get_next_sentence_group = matrix[i - 1][j - 1]
                best_choice = max(add_to_current_sentences_group, get_next_sentence_group)
                if best_choice == add_to_current_sentences_group:
                    actions_matrix[i][j] = self._current_group_flag
                else:
                    actions_matrix[i][j] = self._next_group_flag
                matrix[i][j] = best_choice + sim
        return actions_matrix

    def _get_monotone_matching(self, action_matrix):
        n, m = action_matrix.shape[0] - 1, action_matrix.shape[1] - 1,
        ans = defaultdict()
        i, j = n, m
        while i > 0 and j > 0:
            action = action_matrix[i][j]
            ans[i - 1] = j - 1
            if action == self._current_group_flag:
                i -= 1
            elif action == self._next_group_flag:
                i -= 1
                j -= 1
            else:
                break

        for q in range(1, i):
            ans[q - 1] = j - 1
        for q in range(1, j):
            ans[q - 1] = i - 1

        ans = dict(sorted(ans.items()))
        return ans
