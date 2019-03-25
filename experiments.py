import warnings
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

import tensorflow as tf

from data import MyScaler, hf
from models import GanDaeMLPModel, GDMModelConfig, FewShotConfig, FewShotModel
from metrics import give_metric


warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


def do_experiment(model, data_set, random_state=1000, rf=None):

    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scaler = MyScaler()

    y_true, y_pred, y_score = [], [], []
    for train_index, test_index in fold.split(data_set.x, data_set.y):
        train_set, test_set = data_set.subset(train_index), data_set.subset(test_index)

        train_set.x = scaler.fit_transform(train_set.x)
        test_set.x = scaler.transform(test_set.x)

        model.fit(train_set)
        y_true.append(test_set.y)
        if hasattr(model, 'predict_proba'):
            score = model.predict(test_set)
            y_score.append(score)
            y_pred.append(score > 0.5)
        else:
            y_pred.append(model.predict(test_set))

    if len(y_score) > 0:
        metric = give_metric(y_true, y_pred, y_score, 'Proposed')
    else:
        metric = give_metric(y_true, y_pred, None, 'Few shot')

    print(metric)
    if rf is not None:
        with open(rf, 'a') as wf:
            wf.write(str(metric) + '\n')


def gdm_model_experiment(data_set=None, random_state=1000):
    if data_set is None:
        data_set = hf()

    config = GDMModelConfig()
    config.x_dim = data_set.x_dim
    config.a_dim = data_set.a_dim

    model = GanDaeMLPModel(config)
    do_experiment(model, data_set, random_state)


def search_param(data_set):
    alphas = [1, 0.1, 0.01, 0.001]
    # betas = [10, 1, 0.1, 0.01, 0.001]
    betas = [0]
    l2s = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    act_fns = ['relu', 'sigmoid']

    x_dim = data_set.x_dim
    a_dim = data_set.a_dim

    result_file = "result/gdm/result"

    for alpha in alphas:
        for beta in betas:
            for l2 in l2s:
                for act_fn in act_fns:
                    config = GDMModelConfig()
                    config.alpha = alpha
                    config.beta = beta
                    config.l2 = l2
                    config.act_fn = act_fn
                    config.x_dim = x_dim
                    config.a_dim = a_dim

                    tf.reset_default_graph()
                    gc.collect()
                    model = GanDaeMLPModel(config)
                    with open(result_file, 'a') as wf:
                        wf.write(config.to_json_string())

                    for _ in range(5):
                        do_experiment(model, data_set, None, "result/gdm/result")


def few_shot_experiment(data_set=None, random_state=1000):
    if data_set is None:
        data_set = hf()

    config = FewShotConfig()
    config.x_dim = data_set.x_dim + data_set.a_dim
    config.a_dim = data_set.a_dim

    model = FewShotModel(config)
    do_experiment(model, data_set, random_state)
