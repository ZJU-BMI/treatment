import subprocess

import experiments
from data import hf, acs


if __name__ == "__main__":
    data_sets = ['hf']
    l2_coefficients = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    kernels = ['rbf', 'poly']

    svm_cmd = "python main.py --model svm --dataset {} --regular {} --kernel {}"
    lr_cmd = "python main.py --model lr --dataset {} --regular {}"

    for data_set in data_sets:
        print(data_set)

        # for l2 in l2_coefficients:
        #     cmd = lr_cmd.format(data_set, l2)
        #     print(cmd)
        #     subprocess.call(cmd)
        #
        # for kernel in kernels:
        #     for l2 in l2_coefficients:
        #         cmd = svm_cmd.format(data_set, l2, kernel)
        #         print(cmd)
        #         subprocess.call(cmd)

        if data_set == 'hf':
            s = hf()
        else:
            s = acs()
        experiments.search_param(s)
