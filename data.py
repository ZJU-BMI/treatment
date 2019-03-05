import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class DataSet(object):
    def __init__(self, x, a, y):
        self.x = x
        self.a = a
        self.y = y

    @classmethod
    def load(cls,
             file_path,
             label_column,
             feature_columns,
             treatment_columns):
        df = pd.read_csv(file_path, engine='python', encoding='gbk')
        x = df.iloc[:, feature_columns].values
        y = np.reshape(df[label_column].values, (-1, 1))
        a = df.iloc[:, treatment_columns].values
        return cls(x, a, y)

    @property
    def examples(self):
        return self.x.shape[0]

    @property
    def x_dim(self):
        return self.x.shape[1]

    @property
    def a_dim(self):
        return self.a.shape[1]

    def subset(self, index):
        return DataSet(self.x[index], self.a[index], self.y[index])


class MyScaler(object):
    def __init__(self):
        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.scaler = StandardScaler()

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def fit(self, x):
        self.scaler.fit(x)
        self.imputer.fit(x)

    def transform(self, x):
        result = self.scaler.transform(x)
        result = self.imputer.transform(result)
        return result


def hf():
    return DataSet.load('./resources/心衰_一年再入院_8类.csv',
                        label_column='1年内心源性再住院',
                        feature_columns=[i for i in range(1, 92, 1)],
                        treatment_columns=[i for i in range(92, 106, 1)])


def acs():
    return DataSet.load('./resources/data_processed.csv',
                        label_column='缺血752',
                        feature_columns=[i for i in range(12, 320, 1)],
                        treatment_columns=[i for i in range(320, 338, 1)])
