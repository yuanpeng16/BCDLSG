import os
import argparse
import numpy as np


def load_file(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

    ret = []
    for line in lines:
        terms = line.strip().split()
        ret.append([float(x) for x in terms])

    return np.asarray(ret)


def get_line(line):
    line = line[1:]
    mean = line[1:4]
    std = line[6:9]
    sample_acc = mean[0], std[0]
    set_acc = mean[1], std[1]
    random_acc = mean[2], std[2]
    return sample_acc, set_acc, random_acc


def load(fn):
    data = load_file(fn)
    first = get_line(data[0])
    last = get_line(data[-1])
    return first, last


def format_print(matrix):
    for line in matrix:
        print(' & '.join(line) + ' \\\\')


def dump(matrix, exp_names):
    matrix = np.transpose(matrix, [0, 2, 1, 3])
    assert len(matrix) == len(exp_names)
    ret = []
    for name, experiments in zip(exp_names, matrix):
        line = [name]
        for i, test in enumerate(experiments):
            for term in test:
                mean = str(round(term[0], 1))
                std = str(round(term[1], 1))
                if i == len(experiments) - 1 and len(std) < 4:
                    element = mean + ' {\\small$\\pm$ \\ \\ ' + std + '\\par}'
                else:
                    element = mean + ' {\\small$\\pm$ ' + std + '\\par}'
                line.append(element)
        ret.append(line)
    format_print(ret)


def main(args):
    if args.experiment_set == 'main':
        all_names = [
            ['DNN', 'main_dnn'],
            ['CNN', 'main_cnn'],
            ['ResNet', 'main_resnet'],
            ['ViT', 'main_vit'],
            ['LSTM', 'main_lstm'],
            ['LSTM-1', 'main_lstm-1'],
            ['Transformer', 'main_transformer'],
            ['aPY', 'zeroshot_apy'],
            ['AwA2', 'zeroshot_awa2'],
            ['CUB', 'zeroshot_cub'],
            ['SUN', 'zeroshot_sun'],
            ['DNN', 'equal_dnn'],
            ['CNN', 'equal_cnn'],
            ['Tile', 'label_tile'],
            ['Oneshot', 'label_oneshot'],
            ['Single DNN', 'single_dnn'],
            ['Single CNN', 'single_cnn'],
        ]
    elif args.experiment_set == 'natural':
        all_names = [
            ['DNN', 'review_dnn'],
            ['CNN', 'review_cnn'],
            ['ResNet', 'review_resnet'],
            ['ViT', 'review_vit'],
            ['LSTM', 'review_lstm'],
            ['LSTM-1', 'review_lstm-1'],
            ['Transformer', 'review_transformer'],
            ['64', 'ablation_dnn-64'],
            ['128', 'ablation_dnn-128'],
            ['512', 'ablation_dnn-512'],
            ['1024', 'ablation_dnn-1024'],
            ['Dropout', 'ablation_dnn-dropout'],
            ['Mixup', 'ablation_dnn-mixup'],
        ]
    else:
        raise ValueError(
            '{0} is not a valid experiment_set.'.format(args.experiment_set))
    names = [x[1] for x in all_names]
    exp_names = [x[0] for x in all_names]
    terms = []
    for name in names:
        fn = os.path.join('outputs', name, name + '_acc.txt')
        terms.append(load(fn))
    matrix = np.asarray(terms)
    dump(matrix, exp_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_set', type=str, default='natural',
                        help='Experiment set.')
    main(parser.parse_args())
