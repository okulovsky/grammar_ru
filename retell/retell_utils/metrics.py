import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tg.common.analysis import Bootstrap, Aggregators, grbar_plot


def get_cosine_sim(*strs):
    vectors = [t for t in _get_vectors(*strs)]
    return cosine_similarity(vectors)[0][1]


def _get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def get_jaccard_index(doc1, doc2):
    words_doc_1 = set(doc1.lower().split())
    words_doc_2 = set(doc2.lower().split())
    intersection = words_doc_1.intersection(words_doc_2)
    union = words_doc_1.union(words_doc_2)
    J = float(len(intersection)) / len(union) if len(union) != 0 else 0
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


def compute(df):
    return df.groupby('metric_names').metric_values.mean().to_frame().transpose()


def plot_confint(jaccard_sim, cos_sim, orient='v', ax=None, i=None):
    metrics_names = ["jaccard_sim" for _ in range(len(jaccard_sim))] + ["cos_sim" for _ in range(len(cos_sim))]
    df = pd.DataFrame(data=zip(np.concatenate([jaccard_sim, cos_sim]), metrics_names),
                      columns=['metric_values', 'metric_names'])
    rdf = Bootstrap(df=df, method=compute).run(N=1000)
    rdf_i = rdf[['jaccard_sim', 'cos_sim']].unstack().to_frame().reset_index()
    rdf_i.columns = ['metric_names', 'iteration', 'metric']
    grbar_plot(
        rdf_i.groupby('metric_names').metric.feed(Aggregators.normal_confint()).reset_index(),
        value_column='metric_value',
        error_column='metric_error',
        color_column='metric_names',
        orient=orient,
        ax=None if ax is None else ax[i]
    )


def plot_mutiple_confints(jaccard_sims, cos_sims, orients, figsize=(18, 18)):
    subplot_size = len(jaccard_sims)
    fig, axis = plt.subplots(subplot_size, 1, figsize=figsize)
    for i in range(subplot_size):
        plot_confint(jaccard_sims[i], cos_sims[i], orients[i], ax=axis, i=i)
