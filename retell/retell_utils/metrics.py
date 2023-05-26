import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_cosine_sim(*strs):
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)[0][1]


def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def jaccard_text(doc1, doc2):
    words_doc_1 = set(doc1.lower().split())
    words_doc_2 = set(doc2.lower().split())
    intersection = words_doc_1.intersection(words_doc_2)
    union = words_doc_1.union(words_doc_2)
    J = float(len(intersection)) / len(union)
    return J


def plot_bar_jac_cos_metric(jac, cos):
    fig, axis = plt.subplots(2, 1)
    axis[0].bar(range(len(jac)), jac)
    axis[0].set_title('Индекс Жаккара')
    axis[1].bar(range(len(cos)), cos)
    axis[1].set_title('Косинусное расстояние')
    plt.subplots_adjust(left=0, right=1, wspace=0, hspace=0.5)


def show_statistics_and_bar(jaccard_sim, cos_sim):
    plot_bar_jac_cos_metric(jaccard_sim, cos_sim)
    for name, val in zip(['Индекс Жаккара', 'Косинусное расстояние'], [jaccard_sim, cos_sim]):
        for func_name, func in zip(['median', 'max', 'min'], [np.median, np.max, np.min]):
            print(f"{func_name} {name}: {round(func(val), 3)}")
        print('------------------------------------')
