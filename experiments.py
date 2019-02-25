import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *

from data import MyScaler, eight_hf
from models import BaseModel, ModelConfig


def evaluate(model, test_set):
    y_score = model.predict(test_set)
    y_pred = y_score > 0.5
    acc = accuracy_score(y_true=test_set.y, y_pred=y_pred)
    auc_score = roc_auc_score(y_true=test_set.y, y_score=y_score)
    precision = precision_score(y_true=test_set.y, y_pred=y_pred)
    recall = recall_score(y_true=test_set.y, y_pred=y_pred)
    f1 = f1_score(y_true=test_set.y, y_pred=y_pred)
    return acc, auc_score, precision, recall, f1


def average(metrics):
    m = []
    for metric in zip(*metrics):
        m.append(np.mean(metric))
    return tuple(m)


def do_experiment(data_set=None, random_state=1000):
    if data_set is None:
        data_set = eight_hf

    fold = StratifiedKFold(n_splits=5, random_state=random_state)
    scaler = MyScaler()
    config = ModelConfig()
    model = BaseModel(config)
    metrics = []

    for train_index, test_index in fold.split(data_set.x, data_set.y):
        train_set, test_set = data_set.subset(train_index), data_set.subset(test_index)

        train_set.x = scaler.fit_transform(train_set.x)
        test_set.x = scaler.transform(test_set.x)

        model.fit(train_set)
        metric = evaluate(model, test_set)
        metrics.append(metric)

    print(average(metrics))
