import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, f1_score

from tg.common.ml import batched_training as bt


def plot_confusion_matrix(df, *, labels: list[str]):
    prefix = 'predicted_label_'
    start_idx = df.label.min()
    target = (df.label - start_idx).tolist()
    probas = np.zeros(shape=[len(df), len(labels)])
    for i, (_, row) in enumerate(df.iterrows()):
        for j in range(probas.shape[1]):
            probas[i][j] = row[f'{prefix}{start_idx + j}']

    preds = np.argmax(probas, axis=1).tolist()
    cm = confusion_matrix(target, preds, normalize='true').round(2)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax)


class AlternativeTaskMulticlassMetrics(bt.Metric):
    def __init__(self, label_count: int) -> None:
        super().__init__()
        self.label_count = label_count

    def get_names(self):
        return ['roc_auc', 'f1_weighted']

    def measure(self, df, _):
        prefix = 'predicted_label_'
        start_idx = df.label.min()
        target = (df.label - start_idx).tolist()
        probas = np.zeros(shape=[len(df), self.label_count])

        for i, (_, row) in enumerate(df.iterrows()):
            for j in range(probas.shape[1]):
                probas[i][j] = row[f'{prefix}{start_idx + j}']

        preds = np.argmax(probas, axis=1).tolist()

        result = [
            roc_auc_score(target, probas, multi_class='ovo'),
            f1_score(target, preds, average='weighted')
        ]

        return result
