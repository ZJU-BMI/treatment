import argparse
import warnings

import experiments
import benchmark

from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from data import hf, acs


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_state', type=int, default=500)
    parser.add_argument('--model', type=str, default='proposed',
                        choices=['lr', 'svm', 'rf', 'mlp', 'proposed', 'few_shot'])
    parser.add_argument('--dataset', type=str, default='acs',
                        choices=['hf', 'acs'])
    parser.add_argument('--regular', type=float, default='0.0001')
    parser.add_argument('--kernel', type=str, default='poly',
                        choices=['rbf', 'linear', 'poly'])

    args = parser.parse_args()
    return args


def main():
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
    args = parse_arg()

    if args.dataset == 'acs':
        data_set = acs()
    else:
        data_set = hf()
    if args.model == 'proposed':
        experiments.gdm_model_experiment(data_set, args.random_state)
    elif args.model == 'few_shot':
        experiments.few_shot_experiment(data_set, args.random_state)
    elif args.model == 'lr':
        for _ in range(5):
            benchmark.lr(data_set, None, C=1 / args.regular)  # 每次结果都一样，不需要跑多次
    elif args.model == 'svm':
        for _ in range(5):
            benchmark.svm(data_set, None,
                          C=1 / args.regular,
                          kernel=args.kernel)  # 每次结果都一样，不需要跑多次
    elif args.model == 'rf':
        benchmark.rf(data_set, args.random_state)
    elif args.model == 'mlp':
        pass
    else:
        print('unsupported')


if __name__ == '__main__':
    main()
