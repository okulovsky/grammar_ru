import numpy as np


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


def apk(pred, k=5):
    actual, predicted = pred.label_ids,pred.predictions.argmax(-1)
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
#
#
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if len(actual) == 0:
        return 0.0

    return score / min(len(actual), k)

def mapk(pred, k=5):
    return np.mean([apk(p, k) for p in pred])
