import pandas as pd
import matplotlib.pyplot as plt


auc = pd.read_csv('./result/gdm/AUC.csv', header=None)
auc = auc.values

plt.plot(auc[0], auc[1], label='roc curve of ADTEP')
plt.plot(auc[3], auc[4], label='roc curve of DTEP')
plt.plot(auc[6], auc[7], label='roc curve of LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('roc curve of models (ACS)')
plt.legend(loc='lower right')
plt.show()


plt.plot(auc[11], auc[12], label='roc curve of ADTEP')
plt.plot(auc[14], auc[15], label='roc curve of DTEP')
plt.plot(auc[17], auc[18], label='roc curve of LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('roc curve of models (HF)')
plt.legend(loc='lower right')
plt.show()

