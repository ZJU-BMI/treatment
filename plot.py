import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc as auc_score


auc = pd.read_csv('./result/gdm/AUC.csv', header=None)
auc = auc.values

plt.plot(auc[0], auc[1], label='roc curve of ADTEP')
plt.plot(auc[3], auc[4], label='roc curve of DTEP')
plt.plot(auc[6], auc[7], label='roc curve of LR')
plt.plot(auc[9], auc[10], label='roc curve of SVM')
plt.plot(auc[12], auc[13], label='roc curve of GRACE')
grace_fpr = auc[12][np.logical_not(np.isnan(auc[12]))]
grace_tpr = auc[13][np.logical_not(np.isnan(auc[13]))]
print(auc_score(grace_fpr, grace_tpr))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('roc curve of models (ACS)')
plt.legend(loc='lower right')
plt.show()


plt.plot(auc[17], auc[18], label='roc curve of ADTEP')
plt.plot(auc[20], auc[21], label='roc curve of DETEP')
plt.plot(auc[23], auc[24], label='roc curve of LR')
plt.plot(auc[26], auc[27], label='roc curve of LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('roc curve of models (HF)')
plt.legend(loc='lower right')
plt.show()

