import argparse

import experiments
import benchmark


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_state', type=int, default=1000)
    parser.add_argument('--model', type=str, default='rf',
                        choices=['lr', 'svm', 'rf', 'mlp', 'proposed'])

    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    if args.model == 'proposed':
        experiments.do_experiment(None, args.random_state)
    elif args.model == 'lr':
        benchmark.lr(None, args.random_state)
    elif args.model == 'svm':
        benchmark.svm(None, args.random_state)
    elif args.model == 'rf':
        benchmark.rf(None, args.random_state)
    elif args.model == 'mlp':
        pass
    else:
        print('unsupported')


if __name__ == '__main__':
    main()
