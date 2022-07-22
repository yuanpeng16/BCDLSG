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


def format_print(matrix):
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
    format_print(ret)


def dump(matrix, exp_names):
    type_names = ['Individual', 'Shared']
    exp_names = ['\\multicolumn{1}{c}{' + x + '}' for x in exp_names]
    matrix = np.transpose(matrix, [2, 1, 0, 3])
    for i in range(len(matrix)):
        output_result(matrix[i], type_names, exp_names)
        print()


def dump_vertical(matrix, exp_names):
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


def main():
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
    ]
    names = [x[1] for x in all_names]
    exp_names = [x[0] for x in all_names]
    terms = []
    for name in names:
        fn = os.path.join('outputs', name, name + '_acc.txt')
        terms.append(load(fn))
    matrix = np.asarray(terms)

    # dump(matrix, exp_names)
    dump_vertical(matrix, exp_names)


if __name__ == '__main__':
    main()

# temp = [
#    ['aPY tile', 'zeroshot_apy_tile_cnn_acc'],
#    ['AwA2 tile', 'zeroshot_awa2_tile_cnn_acc'],
#    ['CUB tile', 'zeroshot_cub_tile_cnn_acc'],
#    ['SUN tile', 'zeroshot_sun_tile_cnn_acc']
# ]
