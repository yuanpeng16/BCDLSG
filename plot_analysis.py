import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def read(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    matrix = np.asarray(
        [[float(x) for x in line.strip().split()] for line in lines[:-1]])
    return matrix


def draw(show_legend, lists, stds, legends, basedir, colors, labels, v_name,
         u_name):
    font_size = 24
    x_lim = len(labels)
    plt.figure(figsize=(9, 6))
    ax = plt.subplot(1, 1, 1)
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    for i, (l, s, legend) in enumerate(zip(lists, stds, legends)):
        color_index = min(i, len(colors) - 1)
        color = colors[color_index]
        l1 = np.asarray(l)
        s1 = np.asarray(s)
        ls = '-' if i % 2 == 0 else '--'
        lw = 2
        ax.plot(l1, lw=lw, ls=ls, color=color, label=legend)
        ax.fill_between(np.arange(x_lim), l1 - s1, l1 + s1, color=color,
                        alpha=0.2)

    ax.set_xlim([0, x_lim - 1])
    interval = len(labels) // 4
    xticks = [i for i in range(x_lim) if i % interval == 0]
    labels = [x for i, x in enumerate(labels) if i % interval == 0]
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel(u_name, fontsize=font_size)
    ax.set_ylabel(v_name, fontsize=font_size)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 5

    if show_legend:
        ax.legend(loc='best', prop={'size': font_size})

    plt.savefig(basedir + '.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.clf()


def get_y_label(output_id, size):
    if output_id == size - 2:
        ret = "Partitions"
    elif output_id == size - 1:
        ret = "Accuracy (%)"
    elif output_id % 3 == 2:
        ret = "Ratio (%)"
    else:
        ret = "Partitions"
    return ret


def get_name_list():
    ret = [
        'test_iid_train',
        'test_ood_train',
        'test_ratio_train',
        'test_iid_truth',
        'test_ood_truth',
        'test_ratio_truth',
        'random_iid_train',
        'random_ood_train',
        'random_ratio_train',
        'random_iid_truth',
        'random_ood_truth',
        'random_ratio_truth',
        'train_partitions',
        'train_accuracy'
    ]
    return ret


def main(args):
    font = {'family': 'serif'}
    rc('font', **font)

    experiment_id = args.experiment_id
    log_prefix = os.path.join('logs', experiment_id, experiment_id)
    individual_prefix = log_prefix + '_0_7_'
    shared_prefix = log_prefix + '_7_0_'

    individual_matrix_list = []
    shared_matrix_list = []
    for i in range(1, 6):
        individual_log = os.path.join(individual_prefix + str(i), 'log.txt')
        shared_log = os.path.join(shared_prefix + str(i), 'log.txt')

        individual_matrix_list.append(read(individual_log))
        shared_matrix_list.append(read(shared_log))

    matrix = [individual_matrix_list, shared_matrix_list]
    matrix = np.asarray(matrix)
    matrix = np.transpose(matrix, [3, 0, 2, 1])
    steps, matrix = np.split(matrix, [1], axis=0)
    steps = np.transpose(steps, [0, 1, 3, 2])
    steps = steps[0][0][0].astype(int)

    mean_matrix = np.mean(matrix, axis=-1)
    std_matrix = np.std(matrix, axis=-1)

    output_dir = os.path.join('outputs', experiment_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    name_list = get_name_list()
    legends = ['Individual', 'Shared']
    basedir = os.path.join(output_dir, experiment_id)
    colors = ['blue', 'orange']
    u_name = 'Steps'
    num_columns = len(mean_matrix)
    assert len(name_list) == num_columns
    for i in range(num_columns):
        mean = mean_matrix[i]
        std = std_matrix[i]
        prefix = basedir + "_" + str(i + 1) + "_" + name_list[i]
        v_name = get_y_label(i, num_columns)
        show_legend = i == 12
        draw(show_legend, mean, std, legends, prefix, colors, steps, v_name,
             u_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', type=str,
                        default='partition-t_main_dnn',
                        help='Experiment type.')
    main(parser.parse_args())
