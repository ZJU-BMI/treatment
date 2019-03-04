from sklearn.model_selection import StratifiedKFold

from data import MyScaler, eight_hf
from models import BaseModel, ModelConfig
from metrics import evaluate, average_metric


def do_experiment(data_set=None, random_state=1000):
    if data_set is None:
        data_set = eight_hf

    fold = StratifiedKFold(n_splits=5, random_state=random_state)
    scaler = MyScaler()

    config = ModelConfig()
    config.x_dim = data_set.x_dim
    config.a_dim = data_set.a_dim

    model = BaseModel(config)
    metrics = []

    for train_index, test_index in fold.split(data_set.x, data_set.y):
        train_set, test_set = data_set.subset(train_index), data_set.subset(test_index)

        train_set.x = scaler.fit_transform(train_set.x)
        test_set.x = scaler.transform(test_set.x)

        model.fit(train_set)
        metric = evaluate(model, test_set)
        metrics.append(metric)

    print(average_metric(metrics))
