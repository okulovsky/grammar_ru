import torch
import pandas as pd

def __get_torch_embedding(token, vectorizer, vocab, vec_shape=300):
    if type(vectorizer) is pd.DataFrame:
        return torch.tensor(vectorizer.iloc[vocab[token]][:vec_shape])
    else:
        return vectorizer(torch.tensor(vocab[token]))[:vec_shape]


def get_vocab_embedding(token, vectorizer, vocab, vec_shape=300):
    if token in vocab:
        return __get_torch_embedding(token, vectorizer, vocab, vec_shape)
    else:
        return torch.zeros(vec_shape)


def get_split_tokens_embedding_part(token, vectorizer, vocab, vec_shape=300):
    return get_split_tokens_embedding_full(token, vectorizer, vocab)[:vec_shape]


def get_split_tokens_embedding_full(token, vectorizer, vocab):
    first_item = vectorizer.iloc[0].values if type(vectorizer) is pd.DataFrame \
        else vectorizer(torch.tensor(0))
    full_len = len(first_item)
    token_splited = token.split('-')
    if len(token_splited) == 2 and len(token_splited[0]) > 2:
        if len(token_splited[1]) > 2:
            if token_splited[0] in vocab and token_splited[1] in vocab:
                return sum(__get_torch_embedding(token_splited[i], vectorizer, vocab, full_len)
                           for i in range(2))
        elif token_splited[1] in ["ка", "то"]:
            return __get_torch_embedding(token_splited[0], vectorizer, vocab, full_len)
    return get_vocab_embedding(token, vectorizer, vocab, full_len)
