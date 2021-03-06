import numpy as np
import matplotlib.pyplot as plt

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


def give_metric(y_true, y_pred, y_score=None, model_name=None, result_file=None):
    acc, auc_score, precision, recall, f1 = [], [], [], [], []
    for t, p in zip(y_true, y_pred):
        acc.append(accuracy_score(y_true=t, y_pred=p))
        precision.append(precision_score(y_true=t, y_pred=p))
        recall.append(recall_score(y_true=t, y_pred=p))
        f1.append(f1_score(y_true=t, y_pred=p))

    if y_score is not None:
        fprs, tprs = [], []
        for t, s in zip(y_true, y_score):
            fpr, tpr, _ = roc_curve(y_true=t, y_score=s)
            auc_score.append(auc(fpr, tpr))
            fprs.append(fpr)
            tprs.append(tpr)
        plot_average_roc(fprs, tprs, model_name, "./result/gdm/result")
    else:
        auc_score = [-1 for _ in range(len(y_true))]
    metrics = [acc, auc_score, precision, recall, f1]
    return [np.average(i) for i in metrics], [np.std(i) for i in metrics]


def plot_average_roc(fpr, tpr, model_name, result_file=None):
    all_fpr = np.unique(np.concatenate(fpr))

    mean_tpr = np.zeros_like(all_fpr)
    for f, t in zip(fpr, tpr):
        mean_tpr += np.interp(all_fpr, f, t)

    mean_tpr /= len(fpr)

    for i in range(len(fpr)):
        label = 'auc of cv {}: area = {:.3f}'.format(i, auc(fpr[i], tpr[i]))
        plt.plot(fpr[i], tpr[i], linestyle='-',
                 label=label)

    area = auc(all_fpr, mean_tpr)
    label = 'average auc = {:.3f}'.format(area)
    plt.plot(all_fpr, mean_tpr,
             label=label,
             color='deeppink',
             linestyle=':')

    if result_file is not None:
        with open(result_file, 'a') as af:
            all_fpr_str = np.array2string(all_fpr, precision=4, separator=',').replace('\n', '').replace(" ", "") + '\n'
            mean_tpr_str = np.array2string(mean_tpr, precision=4, separator=',').replace('\n', '').replace(" ", "") + '\n'
            af.write(all_fpr_str)
            af.write(mean_tpr_str)
            af.write('\n')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('roc curve for model : {}'.format(model_name))
    plt.legend(loc='lower right')
    plt.show()
