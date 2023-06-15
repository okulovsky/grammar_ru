from tqdm.notebook import tqdm
import string, torch
import numpy as np


def __get_sorted_sentences_importance(importances, sentences, retell_detail, sentences_sep=" "):
    sorted_index_array = np.argsort(importances)[::-1]
    top_sentences = [sentences_sep.join(sentences[id]) for id in sorted_index_array[:retell_detail]]
    return top_sentences


def get_extract_retell(texts, sentence_extractor, embeding_func, vectorizer, vocab, retell_detail=5,
                       ban_words=string.punctuation):
    for text in texts[:1]:  #  не только первую книгу и тд, мб передавать генератор
        extract_retell = []
        for chapter in tqdm(text.index):
            sentences = sentence_extractor(chapter)
            vec_sum_sentences = []
            for sentence in sentences:
                tokens = [word.lower() for word in sentence if word not in ban_words]
                sentence_vector_sum = sum([embeding_func(token, vectorizer, vocab) for token in tokens])
                vec_sum_sentences.append(sentence_vector_sum / len(tokens) if len(tokens) > 0 else torch.zeros(300))
            importances = []
            cos = torch.nn.CosineSimilarity(dim=0)
            for i in range(len(vec_sum_sentences)):
                importance = 0
                for j in range(i, len(vec_sum_sentences)):
                    importance += cos(vec_sum_sentences[i], vec_sum_sentences[j])
                importances.append(importance)
            top_sentences = __get_sorted_sentences_importance(importances, sentences, retell_detail)
            extract_retell.append("\n".join(top_sentences))
        return extract_retell  #  Если будет несколько книг, нужно сдвинуть return до первого for-а

## Здесь нужно использовать эмбедер, который даёт полный вектор, хз пока как это сделать
def better_get_extract_retell__(texts, sentence_extractor, embeding_func, vectorizer, vocab, retell_detail=5,
                                ban_words=string.punctuation):
    for text in texts[:1]:
        extract_retell = []
        for chapter in tqdm(text.index):
            sentences = sentence_extractor(chapter)
            vec_sum_sentences_f = []
            vec_sum_sentences_s = []
            for sentence in sentences:
                tokens = [word.lower() for word in sentence if word not in ban_words]
                embeded = [embeding_func(token, vectorizer, vocab) for token in tokens]
                sentence_vector_sum_f = sum([vec[:300] if len(vec) > 300 else vec for vec in embeded])
                sentence_vector_sum_s = sum([vec[301:601] if len(vec) > 300 else vec for vec in embeded])
                vec_sum_sentences_f.append(sentence_vector_sum_f / len(tokens) if len(tokens) > 0 else torch.zeros(300))
                vec_sum_sentences_s.append(sentence_vector_sum_s / len(tokens) if len(tokens) > 0 else torch.zeros(300))
            importances = []
            cos = torch.nn.CosineSimilarity(dim=0)
            for i in range(len(vec_sum_sentences_f)):
                importance = 0
                for j in range(len(vec_sum_sentences_f)):
                    if i != j:
                        importance += cos(vec_sum_sentences_f[i], vec_sum_sentences_s[j])
                importances.append(importance)
            top_sentences = __get_sorted_sentences_importance(importances, sentences, retell_detail)
            extract_retell.append("\n".join(top_sentences))
        return extract_retell  #  Если будет несколько книг, нужно сдвинуть return до первого for-а
