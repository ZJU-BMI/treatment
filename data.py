import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class HF(object):
    def __init__(self, x, a, y):
        self.x = x
        self.a = a
        self.y = y

    @classmethod
    def load(cls,
             file_path="./resources/心衰_一年再入院.csv",
             label_column='一年内心源性再住院',
             treatment_column='药物分类治疗',
             a_nums=3):
        df = pd.read_csv(file_path, engine='python', encoding='gbk')
        x = df.iloc[:, 1:92].values
        y = np.reshape(df[label_column].values, (-1, 1))
        a = df[treatment_column].values
        a = tf.keras.utils.to_categorical(a, a_nums)
        return cls(x, a, y)

    @property
    def examples(self):
        return self.x.shape[0]

    def subset(self, index):
        return HF(self.x[index], self.a[index], self.y[index])


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


eight_hf = HF.load('./resources/心衰_一年再入院_8类.csv',
                   label_column='1年内心源性再住院',
                   treatment_column='treatment_class',
                   a_nums=8)
