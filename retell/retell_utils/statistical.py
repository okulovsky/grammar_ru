import numpy as np
import string
from collections import Counter
from tqdm.notebook import tqdm


def __get_sorted_sentences_importance(importances, sentences, retell_detail, sentences_sep=" "):
    sorted_index_array = np.argsort(importances)[::-1]
    top_sentences = [sentences_sep.join(sentences[id]) for id in sorted_index_array[:retell_detail]]
    return top_sentences


def get_extract_retell_by_common_word(texts, sentences_and_norm_form_extractor, retell_detail=5,
                                      ban_words=string.punctuation):
    extract_retell = []
    for text in texts:
        for chapter in tqdm(text.index):
            sentences, norm_form_sentences = sentences_and_norm_form_extractor(chapter)
            sentence_score = []
            for i, first_sentence in enumerate(norm_form_sentences):
                coef = 0
                for second_sentence in norm_form_sentences[i + 1:]:
                    words_1 = set(first_sentence) - ban_words
                    words_2 = Counter(second_sentence)
                    coef += sum(words_2[word] for word in words_1) / (len(first_sentence) + len(second_sentence))
                sentence_score.append(coef)
            top_sentences = __get_sorted_sentences_importance(sentence_score, sentences, retell_detail)
            extract_retell.append("\n".join(top_sentences))
        return extract_retell
