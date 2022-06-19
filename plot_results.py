import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def draw(args, lists, stds, legends, basedir, colors, lw, loc, labels, v_name,
         u_name, plot=True, font_size = 24):
    x_lim = len(lists[0])
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
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    for i, (l, s, legend) in enumerate(zip(lists, stds, legends)):
        color_index = min(i, len(colors) - 1)
        entries = colors[color_index]
        color, marker = entries
        l1 = np.asarray(l)
        s1 = np.asarray(s)
        if i % 2 == 0:
            ls = '-'
        else:
            ls = '--'
        ax.plot(l1, lw=lw, markevery=(0, 1),
                ls=ls,
                marker=marker, markersize=16, markeredgewidth=2,
                markerfacecolor='none', color=color, label=legend)
        ax.fill_between(np.arange(x_lim), l1 - s1, l1 + s1,
                        color=color, alpha=0.2)

    ax.set_xlim([0, x_lim - 1])
    ax.set_xticks(range(x_lim))
    legend_font_size = 18
    ax.legend(loc=loc, prop={'size': legend_font_size})
    ax.set_xlabel(u_name, fontsize=font_size)
    ax.set_ylabel(v_name, fontsize=font_size)
    ax.set_xticklabels(labels)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 5
    plt.savefig(basedir + '.pdf', bbox_inches='tight', pad_inches=0.01)


def get_list(lines, key):
    return [100 * float(x[key]) for x in lines]


def get_list_index(lines):
    ret = []
    for x in lines:
        if x[0] == 'final':
            ret.append(-1)
        else:
            ret.append(int(x[0]))
    return ret


def load(fn, steps=False):
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
        lines = [x.strip().split(' ') for x in lines]
        index = [3, 8, 9, 14]
        steps = get_list_index(lines)
        eval1 = get_list(lines, index[0])
        eval2 = get_list(lines, index[1])
        eval3 = get_list(lines, index[2])
        eval4 = get_list(lines, index[3])
    else:
        eval1 = []
        eval2 = []
        eval3 = []
        eval4 = []
    return eval1, eval2, eval3, eval4, steps


def get_results(args, path):
    if args.first_experiment:
        exp_ids = ['1']
    else:
        exp_ids = ['1', '2', '3', '4', '5']
    results = [[], [], [], []]
    for e in exp_ids:
        fn = os.path.join(path + e, "log.txt")
        eval1, eval2, eval3, eval4, _ = load(fn)
        results[0].append(eval1[-1])
        results[1].append(eval2[-1])
        results[2].append(eval3[-1])
        results[3].append(eval4[-1])

    means = []
    stds = []
    for result in results:
        matrix = np.asarray(result)
        means.append(np.mean(matrix, axis=0))
        stds.append(np.std(matrix, axis=0))

    return means, stds


def get_params(args):
    if args.experiment_type == 'main':
        pairs = [
            ('IID Acc', ('b', 'v')),
            ('OOD Acc', ('c', '^')),
            ('OOD Area', ('r', 's')),
            ('ALL Area', ('brown', 'D')),
        ]
        ids = [
            str(i) + '_' + str(args.depth - i) for i in range(args.depth + 1)]
        labels = [
            str(i) + '-' + str(args.depth - i) for i in range(args.depth + 1)]
        eid = args.experiment_id + '_'
        file_list = ['logs/' + eid + c + '_' for c in ids]
        legends = [x[0] for x in pairs]
        colors = [x[1] for x in pairs]
        out_name = eid + 'acc'
        output_list = [
            'outputs/' + out_name + '/' + out_name
        ]
        lw = 2
        loc = 'right'
    elif args.experiment_type == 'steps':
        pairs = [
            ('IID Acc', ('b', 'v')),
            ('OOD Acc', ('c', '^')),
            ('OOD Area', ('r', 's')),
            ('ALL Area', ('brown', 'D')),
        ]
        ids = [
            str(i) + '_' + str(args.depth - i) for i in range(args.depth + 1)]
        labels = [
            str(i) + '-' + str(args.depth - i) for i in range(args.depth + 1)]
        eid = args.experiment_id + '_'
        file_list = ['logs/' + eid + c + '_' for c in ids]
        legends = [x[0] for x in pairs]
        colors = [x[1] for x in pairs]
        out_name = eid + 'acc'
        output_list = [
            'outputs/' + out_name + '/' + out_name + '_steps'
        ]
        lw = 2
        loc = 'upper right'
    else:
        print(args.experiment_type + " is not defined.")
        assert False
    return file_list, legends, output_list, colors, lw, loc, labels


def final_main(args):
    file_list, legends, output_list, colors, lw, loc, labels = get_params(args)
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

    acc_mean = [eval2_list, eval3_list, eval4_list]
    acc_std = [std2_list, std3_list, std4_list]
    draw(args, acc_mean, acc_std, legends, output_list[0], colors, lw, loc,
         labels, 'Accuracy (%)', 'Common-Individual Layer Depth')


def get_steps(args, path):
    if args.first_experiment:
        exp_ids = ['1']
    else:
        exp_ids = ['1', '2', '3', '4', '5']
    results = [[], [], [], []]
    length = 11
    for e in exp_ids:
        fn = os.path.join(path + e, "log.txt")
        eval1, eval2, eval3, eval4, steps = load(fn)
        results[0].append(eval1[1:length])
        results[1].append(eval2[1:length])
        results[2].append(eval3[1:length])
        results[3].append(eval4[1:length])
    steps = steps[1:length]

    means = []
    stds = []
    for result in results:
        matrix = np.asarray(result)
        means.append(np.mean(matrix, axis=0))
        stds.append(np.std(matrix, axis=0))

    return means, stds, steps


def step_main(args):
    file_list, legends, output_list, colors, lw, loc, labels = get_params(args)

    fn = file_list[-1]
    means, stds, steps = get_steps(args, fn)
    eval1, eval2, eval3, eval4 = means
    std1, std2, std3, std4 = stds

    font = {'family': 'serif'}
    rc('font', **font)

    acc_mean = [eval2, eval3, eval4]
    acc_std = [std2, std3, std4]
    draw(args, acc_mean, acc_std, legends, output_list[0], colors, lw, loc,
         steps, 'Accuracy (%)', 'Training Steps', font_size=18)


def main(args):
    if args.experiment_type == 'main':
        final_main(args)
    else:
        step_main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_type', type=str, default='main',
                        help='Experiment type.')
    parser.add_argument('--experiment_id', type=str,
                        default='fashion_mnist_added_diagonal',
                        help='Experiment type.')
    parser.add_argument('--depth', type=int, default=7,
                        help='Depth of the network.')
    parser.add_argument('--analysis', action='store_true', default=False,
                        help='Analysis.')
    parser.add_argument('--first_experiment', action='store_true',
                        default=False,
                        help='Visualize first experiment.')
    parser.add_argument('--random_threshold', type=str, default='75',
                        help='Threshold to randomize the second input.')
    main(parser.parse_args())
