from sklearn.model_selection import StratifiedKFold

from data import MyScaler, hf, acs
from models import GanDaeMLPModel, GDMModelConfig
from metrics import give_metric


def do_experiment(data_set=None, random_state=1000):
    if data_set is None:
        data_set = hf()

    fold = StratifiedKFold(n_splits=5, random_state=random_state)
    scaler = MyScaler()

    config = GDMModelConfig()
    config.x_dim = data_set.x_dim
    config.a_dim = data_set.a_dim

    model = GanDaeMLPModel(config)

    y_true, y_pred, y_score = [], [], []
    for train_index, test_index in fold.split(data_set.x, data_set.y):
        train_set, test_set = data_set.subset(train_index), data_set.subset(test_index)

        train_set.x = scaler.fit_transform(train_set.x)
        test_set.x = scaler.transform(test_set.x)

        model.fit(train_set)
        y_true.append(test_set.y)
        score = model.predict(test_set)
        y_score.append(score)
        y_pred.append(score > 0.5)

    print(give_metric(y_true, y_pred, y_score, 'Proposed'))
