from typing import Dict
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from tg.grammar_ru import SeparatorRu
from tg.grammar_ru.common.separator import AbstractSeparator
from collections import defaultdict


class Seq2VecConverter:
    def __init__(self, model_name='distiluse-base-multilingual-cased', separator: AbstractSeparator = SeparatorRu):
        self.model = SentenceTransformer(model_name)
        self.viewer = separator.Viewer()

    def get_matches(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> Dict[str, str]:
        text_1, text_2 = [self.viewer.to_sentences_strings(df).values for df in [df_1, df_2]]
        encoded_text_1, encoded_text_2 = [self.model.encode(text, convert_to_numpy=True) for text in [text_1, text_2]]
        cos_sim = util.cos_sim(encoded_text_1, encoded_text_2)
        matching = self._get_monotone_matchings(cos_sim)
        matched_sent = {text_1[t_1_id]: text_2[t_2_id] for t_1_id, t_2_id in matching.items()}
        return matched_sent

    def _get_monotone_matchings(self, cos_sim) -> Dict[int, int]:
        n, m = cos_sim.shape[0] + 1, cos_sim.shape[1] + 1
        if n < m:
            n, m = m, n
            cos_sim = cos_sim.T
        matrix = np.zeros((n, m))
        for j in range(m):
            matrix[j][j] = cos_sim[j - 1][j - 1].item()
        for i in range(1, n):  # text
            for j in range(1, min(i, m)):  # retell
                sim = cos_sim[i - 1][j - 1].item()  # -1 в индексе из-за того, что матрица ответа обрамлена нулями
                add_to_current_sentences_group = matrix[i - 1][j]
                get_next_sentence_group = matrix[i - 1][j - 1]
                best_choice = max(add_to_current_sentences_group, get_next_sentence_group)
                matrix[i][j] = best_choice + sim
        ans = self._recover_ans(matrix)
        return ans

    def _recover_ans(self, matrix) -> Dict[int, int]:
        n, m = matrix.shape[0] - 1, matrix.shape[1] - 1,
        ans = defaultdict()
        i, j = n, m
        while i > 1 and j > 1:
            add_to_current_sentences_group = matrix[i - 1, j]
            get_next_sentence_group = matrix[i - 1, j - 1]
            best_choice = min(add_to_current_sentences_group, get_next_sentence_group)
            ans[i - 1] = j - 1
            if best_choice == add_to_current_sentences_group:
                i -= 1
            else:
                i -= 1
                j -= 1

        for q in range(1, i):
            ans[q - 1] = j - 1
        for q in range(1, j):
            ans[q - 1] = i - 1

        ans = dict(sorted(ans.items()))
        return ans
