import numpy as np
from sklearn.metrics import *


def evaluate(model, test_set):
    y_score = model.predict(test_set)
    y_pred = y_score > 0.5
    acc = accuracy_score(y_true=test_set.y, y_pred=y_pred)
    auc_score = roc_auc_score(y_true=test_set.y, y_score=y_score)
    precision = precision_score(y_true=test_set.y, y_pred=y_pred)
    recall = recall_score(y_true=test_set.y, y_pred=y_pred)
    f1 = f1_score(y_true=test_set.y, y_pred=y_pred)
    return acc, auc_score, precision, recall, f1


def average_metric(metrics):
    m = []
    s = []
    for metric in zip(*metrics):
        m.append(np.mean(metric))
        s.append(np.std(metric))

    return m, s
