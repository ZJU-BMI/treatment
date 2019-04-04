from scipy.stats import chi2_contingency
import pandas as pd


s = pd.read_csv('./result/gdm/status.csv', header=None)

for _, v in s.iterrows():
    ob = v.values.reshape([2, 2]).T
    _, p, _, _ = chi2_contingency(ob)
    print(p)
