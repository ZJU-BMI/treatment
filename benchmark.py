from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import numpy as np

from data import hf, MyScaler
from metrics import give_metric


def evaluate(model, test_set):
    y_pred = model.predict(np.concatenate((test_set.x, test_set.a), -1))
    acc = accuracy_score(y_true=test_set.y, y_pred=y_pred)
    precision = precision_score(y_true=test_set.y, y_pred=y_pred)
    recall = recall_score(y_true=test_set.y, y_pred=y_pred)
    f1 = f1_score(y_true=test_set.y, y_pred=y_pred)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(np.concatenate((test_set.x, test_set.a), -1))
        y_score = y_score[:, 1]
        auc_score = roc_auc_score(y_true=test_set.y, y_score=y_score)
    else:
        auc_score = -1

    return acc, auc_score, precision, recall, f1


def experiment(model_class, data_set=None, model_name=None, random_state=1000):
    if data_set is None:
        data_set = hf()
    data_set.y = np.reshape(data_set.y, (-1))

    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scaler = MyScaler()

    y_true, y_pred, y_score = [], [], []
    for train_index, test_index in fold.split(data_set.x, data_set.y):
        train_set, test_set = data_set.subset(train_index), data_set.subset(test_index)

        train_set.x = scaler.fit_transform(train_set.x)
        test_set.x = scaler.transform(test_set.x)

        model = model_class()
        model.fit(np.concatenate((train_set.x, train_set.a), -1), train_set.y)

        y_true.append(test_set.y)
        y_pred.append(model.predict(np.concatenate((test_set.x, test_set.a), -1)))
        if hasattr(model, 'predict_proba'):
            score = model.predict_proba(np.concatenate((test_set.x, test_set.a), -1))
            score = score[:, 1]
            y_score.append(score)

    if len(y_score) > 0:
        print(give_metric(y_true, y_pred, y_score, model_name))
    else:
        print(give_metric(y_true, y_pred, None, model_name))


def lr(data_set=None, random_state=1000, C=1):
    def LRWrapper():
        return LogisticRegression(solver='lbfgs', C=C)
    experiment(LRWrapper, data_set, "Logistic Regression", random_state)


def svm(data_set=None, random_state=1000, C=1, kernel='rbf'):
    def SVCWrapper():
        return SVC(gamma='scale', C=C, kernel=kernel, probability=True)
    experiment(SVCWrapper, data_set, 'SVM', random_state)


def rf(data_set=None, random_state=1000):
    def RandomForestWrapper():
        return RandomForestClassifier(n_estimators=100)
    experiment(RandomForestWrapper, data_set, 'Random Forest', random_state)
