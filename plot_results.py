import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def draw(args, lists, stds, legends, basedir, colors, lw, loc, v_name,
         plot=True):
    x_lim = len(lists[0])
    if x_lim >= 1000:
        marker_scale = 10
        start_scale = marker_scale
    elif args.experiment_type == 'scan':
        marker_scale = 1
        start_scale = 2
    else:
        marker_scale = 2
        start_scale = marker_scale
    directory = os.path.dirname(basedir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(basedir + '.txt', 'w') as f:
        for i in range(len(lists[0])):
            f.write(str(i + 1))
            for e in lists:
                f.write('\t' + str(e[i]))
            for e in stds:
                f.write('\t' + str(e[i]))
            f.write('\n')

    if not plot:
        return

    plt.figure(figsize=(9, 6))
    ax = plt.subplot(1, 1, 1)
    font_size = 24
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    for i, (l, s, legend) in enumerate(zip(lists, stds, legends)):
        color_index = min(i, len(colors) - 1)
        entries = colors[color_index]
        color, marker = entries
        l1 = [0]
        l1.extend(l)
        l1 = np.asarray(l1)
        s1 = [0]
        s1.extend(s)
        s1 = np.asarray(s1)
        if i % 2 == 0:
            ls = '-'
        else:
            ls = '--'
        ax.plot(l1, lw=lw, markevery=(start_scale * 10, marker_scale * 20),
                ls=ls,
                marker=marker, markersize=16, markeredgewidth=2,
                markerfacecolor='none', color=color, label=legend)
        ax.fill_between(np.arange(x_lim + 1), l1 - s1, l1 + s1,
                        color=color, alpha=0.2)

    ax.set_xlim([1, x_lim])
    legend_font_size = 18
    ax.legend(loc=loc, prop={'size': legend_font_size})
    ax.set_xlabel('Steps', fontsize=font_size)
    ax.set_ylabel(v_name, fontsize=font_size)
    ax.xaxis.labelpad = 0
    if args.experiment_type == 'scan':
        ax.yaxis.labelpad = 10
    else:
        ax.yaxis.labelpad = 0
    plt.savefig(basedir + '.pdf', bbox_inches='tight', pad_inches=0.01)


def get_list(lines, key):
    return [float(x[key]) for x in lines]
    ret = []
    for line in lines:
        if key not in line:
            continue
        str_value = line.strip().split(',')[3][1:-1]
        ret.append(100 * float(str_value))
    return ret


def load(fn, scan_output):
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
        lines = [x.strip().split(' ') for x in lines]
        if scan_output:
            indice = [3, 4, 5, 6]
        else:
            indice = [2, 3, 4, 5]
        eval1 = get_list(lines, indice[0])
        eval2 = get_list(lines, indice[1])
        eval3 = get_list(lines, indice[2])
        eval4 = get_list(lines, indice[3])
    else:
        eval1 = []
        eval2 = []
        eval3 = []
        eval4 = []
    return eval1, eval2, eval3, eval4


def get_results(args, path):
    if args.first_experiment:
        exps = ['A']
    else:
        exps = ['A', 'B', 'C', 'D', 'E']

    results = [[], [], [], []]
    for e in exps:
        fn = os.path.join(path + e, "log.txt")
        eval1, eval2, eval3, eval4 = load(fn, args.experiment_type == 'scan')
        if args.experiment_type == 'scan':
            limit = 50
            eval1 = eval1[:limit]
            eval2 = eval2[:limit]
            eval3 = eval3[:limit]
            eval4 = eval4[:limit]
        results[0].append(eval1)
        results[1].append(eval2)
        results[2].append(eval3)
        results[3].append(eval4)

    for r in results[0]:
        assert len(results[0][0]) == len(r)

    means = []
    stds = []
    for result in results:
        matrix = np.asarray(result)
        means.append(np.mean(matrix, axis=0))
        stds.append(np.std(matrix, axis=0))

    return means, stds


def get_params(args):
    if args.experiment_type == 'scan':
        pairs = [
            ('Train A', 'logs/scan_first_large_', ('b', 'v')),
            ('Test A', 'logs/scan_first_large_', ('c', '^')),
            ('Train B', 'logs/scan_second_large_', ('r', 's')),
            ('Test B', 'logs/scan_second_large_', ('brown', 'D')),
        ]
        file_list = [pairs[0][1], pairs[2][1]]
        legends = [x[0] for x in pairs]
        colors = [x[2] for x in pairs]
        output_list = [
            'outputs/scan_large/scan_loss',
            'outputs/scan_large/scan_acc'
        ]
        lw = 2
        loc = 'upper left'
    elif args.experiment_type == 'mnist':
        pairs = [
            ('Train A', 'logs/mnist_first_', ('b', 'v')),
            ('Test A', 'logs/mnist_first_', ('c', '^')),
            ('Train B', 'logs/mnist_second_', ('r', 's')),
            ('Test B', 'logs/mnist_second_', ('brown', 'D')),
        ]
        file_list = [pairs[0][1], pairs[2][1]]
        legends = [x[0] for x in pairs]
        colors = [x[2] for x in pairs]
        output_list = [
            'outputs/mnist/mnist_loss',
            'outputs/mnist/mnist_acc'
        ]
        lw = 2
        loc = 'lower right'
    elif args.experiment_type == 'mnist_paired':
        pairs = [
            ('Train A', 'logs/mnist_paired_first_', ('b', 'v')),
            ('Test A', 'logs/mnist_paired_first_', ('c', '^')),
            ('Train B', 'logs/mnist_paired_second_', ('r', 's')),
            ('Test B', 'logs/mnist_paired_second_', ('brown', 'D')),
        ]
        file_list = [pairs[0][1], pairs[2][1]]
        legends = [x[0] for x in pairs]
        colors = [x[2] for x in pairs]
        output_list = [
            'outputs/mnist_paired/mnist_paired_loss',
            'outputs/mnist_paired/mnist_paired_acc'
        ]
        lw = 2
        loc = 'lower right'
    elif args.experiment_type == 'mnist_threshold':
        thresh = args.random_threshold
        pairs = [
            ('Train A', 'logs/mnist_first_' + thresh + '_', ('b', 'v')),
            ('Test A', 'logs/mnist_first_' + thresh + '_', ('c', '^')),
            ('Train B', 'logs/mnist_second_' + thresh + '_', ('r', 's')),
            ('Test B', 'logs/mnist_second_' + thresh + '_',
             ('brown', 'D')),
        ]
        file_list = [pairs[0][1], pairs[2][1]]
        legends = [x[0] for x in pairs]
        colors = [x[2] for x in pairs]
        output_list = [
            'outputs/mnist_' + thresh + '/mnist_' + thresh + '_loss',
            'outputs/mnist_' + thresh + '/mnist_' + thresh + '_acc'
        ]
        lw = 2
        loc = 'lower right'
    else:
        print(args.experiment_type + " is not defined.")
        assert False
    return file_list, legends, output_list, colors, lw, loc


def main(args):
    file_list, legends, output_list, colors, lw, loc = get_params(args)
    eval1_list = []
    eval2_list = []
    eval3_list = []
    eval4_list = []
    std1_list = []
    std2_list = []
    std3_list = []
    std4_list = []
    for fn in file_list:
        means, stds = get_results(args, fn)
        eval1, eval2, eval3, eval4 = means
        std1, std2, std3, std4 = stds

        eval1_list.append(eval1)
        eval2_list.append(eval2)
        eval3_list.append(eval3)
        eval4_list.append(eval4)
        std1_list.append(std1)
        std2_list.append(std2)
        std3_list.append(std3)
        std4_list.append(std4)

    font = {'family': 'serif'}
    rc('font', **font)

    if args.experiment_type == 'scan':
        for i in range(len(output_list)):
            output_list[i] = output_list[i] + '_trancate'

    acc_mean = [100 * eval2_list[0], 100 * eval4_list[0], 100 * eval2_list[1],
                100 * eval4_list[1]]
    acc_std = [100 * std2_list[0], 100 * std4_list[0], 100 * std2_list[1],
               100 * std4_list[1]]
    draw(args, acc_mean, acc_std, legends, output_list[1], colors, lw, loc,
         'Accuracy (%)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Continual Learning evaluation.')
    parser.add_argument('--experiment_type', type=str, default='mnist_paired',
                        help='Experiment type.')
    parser.add_argument('--analysis', action='store_true', default=False,
                        help='Analysis.')
    parser.add_argument('--first_experiment', action='store_true',
                        default=False,
                        help='Visualize first experiment.')
    parser.add_argument('--random_threshold', type=str, default='75',
                        help='Threshold to randomize the second input.')
    args = parser.parse_args()
    main(args)
