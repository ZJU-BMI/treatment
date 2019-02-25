import argparse

import experiments


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_state', type=int, default=1000)

    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    experiments.do_experiment(None, args.random_state)


if __name__ == '__main__':
    main()
