import os
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
    mean = line[:3]
    std = line[3:]
    sample_acc = mean[0], std[0]
    set_acc = mean[1], std[1]
    random_acc = mean[2], std[2]
    return sample_acc, set_acc, random_acc


def load(fn):
    data = load_file(fn)
    first = get_line(data[0])
    last = get_line(data[-1])
    return first, last


def format(matrix):
    for line in matrix:
        print(' & '.join(line) + ' \\\\')


def output_result(matrix, type_names, exp_names):
    assert len(type_names) == len(matrix)
    ret = [[''] + exp_names]
    for name, vec in zip(type_names, matrix):
        line = [name]
        for term in vec:
            mean = str(round(term[0], 1))
            std = str(round(term[1], 1))
            element = mean + ' {\\small$\\pm$ ' + std + '\\par}'
            line.append(element)
        ret.append(line)
    format(ret)


def main():
    names = [
        'fashion_mnist_added_diagonal_acc',
        'cifar_fashion_added_diagonal_acc',
        'cifar_fashion_added_diagonal_resnet_acc',
        'cifar_fashion_added_diagonal_vit_acc',
        'reuters_reuters_diagonal_lstm_acc',
        'reuters_reuters_diagonal_transformer_acc'
    ]
    terms = []
    for name in names:
        fn = os.path.join('outputs', name, name + '.txt')
        terms.append(load(fn))
    matrix = np.asarray(terms)
    matrix = np.transpose(matrix, [2, 1, 0, 3])

    type_names = ['Shared', 'Individual']
    exp_names = ['DNN', 'CNN', 'ResNet', 'ViT', 'LSTM', 'Transformer']
    exp_names = ['\\multicolumn{1}{c}{' + x + '}' for x in exp_names]
    for i in range(len(matrix)):
        output_result(matrix[i], type_names, exp_names)
        print()


if __name__ == '__main__':
    main()
