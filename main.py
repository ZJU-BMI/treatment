import argparse

import experiments
import benchmark

from data import hf, acs


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_state', type=int, default=1000)
    parser.add_argument('--model', type=str, default='rf',
                        choices=['lr', 'svm', 'rf', 'mlp', 'proposed'])
    parser.add_argument('--dataset', type=str, default='acs',
                        choices=['hf', 'acs'])

    args = parser.parse_args()
    return args


def main():
    args = parse_arg()

    if args.dataset == 'acs':
        data_set = acs()
    else:
        data_set = hf()
    if args.model == 'proposed':
        experiments.do_experiment(data_set, args.random_state)
    elif args.model == 'lr':
        benchmark.lr(data_set, args.random_state)
    elif args.model == 'svm':
        benchmark.svm(data_set, args.random_state)
    elif args.model == 'rf':
        benchmark.rf(data_set, args.random_state)
    elif args.model == 'mlp':
        pass
    else:
        print('unsupported')


if __name__ == '__main__':
    main()
