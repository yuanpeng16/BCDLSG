import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def draw(args, lists, stds, legends, basedir, colors, lw, loc, labels, v_name,
         u_name, plot=True, font_size=24):
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
    if args.show_legend:
        legend_font_size = font_size
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
            ret.append(x[0])
    return ret


def load(fn, steps=False):
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
        lines = [x.strip().split(' ') for x in lines]
        index = [3, 8, 9, 14]
        steps = get_list_index(lines)
        eval_list = [get_list(lines, i) for i in index]
    else:
        eval_list = [[], [], [], []]
    return eval_list, steps


def get_results(args, path):
    if args.first_experiment:
        exp_ids = ['1']
    else:
        exp_ids = ['1', '2', '3', '4', '5']
    results = [[], [], [], []]
    for e in exp_ids:
        fn = os.path.join(path + e, "log.txt")
        eval_list, _ = load(fn)
        assert len(results) == len(eval_list)
        for result, evaluation in zip(results, eval_list):
            result.append(evaluation[-1])

    means = []
    stds = []
    for result in results:
        matrix = np.asarray(result)
        means.append(np.mean(matrix, axis=0))
        stds.append(np.std(matrix, axis=0))

    return means, stds


def get_params(args):
    if args.experiment_id == 'main_resnet':
        depth = 5
    elif args.experiment_id == 'main_lstm_shallow':
        depth = 2
    else:
        depth = 7

    if args.experiment_type == 'main':
        pairs = [
            ('Eval Sample Acc.', ('b', 'v')),
            ('Test Sample Acc.', ('c', '^')),
            ('Test Set Acc.', ('r', 's')),
            ('Rnd. Set Acc.', ('brown', 'D')),
        ]
        ids = [
            str(i) + '_' + str(depth - i) for i in range(depth + 1)]
        labels = [
            str(i) + '-' + str(depth - i) for i in range(depth + 1)]
        eid = args.experiment_id + '_'
        log_dir = 'logs/' + args.experiment_id + '/'
        file_list = [log_dir + eid + c + '_' for c in ids]
        legends = [x[0] for x in pairs]
        colors = [x[1] for x in pairs]
        out_name = eid + 'acc'
        output_list = [
            'outputs/' + args.experiment_id + '/' + out_name
        ]
        lw = 2
        loc = 'upper right'
    elif args.experiment_type == 'steps':
        pairs = [
            ('OOD Area', ('b', '')),
            ('Random Area', ('r', '')),
        ]
        ids = [
            str(i) + '_' + str(depth - i) for i in range(depth + 1)]
        labels = [
            str(i) + '-' + str(depth - i) for i in range(depth + 1)]
        eid = args.experiment_id + '_'
        log_dir = 'logs/' + args.experiment_id + '/'
        file_list = [log_dir + eid + c + '_' for c in ids]
        legends = [x[0] for x in pairs]
        colors = [x[1] for x in pairs]
        out_name = eid + 'acc'
        output_list = [
            'outputs/' + args.experiment_id + '/' + out_name + '_steps'
        ]
        lw = 2
        loc = 'upper right'
    else:
        print(args.experiment_type + " is not defined.")
        assert False
    return file_list, legends, output_list, colors, lw, loc, labels


def final_main(args):
    file_list, legends, output_list, colors, lw, loc, labels = get_params(args)
    eval_lists = [[], [], [], []]
    std_lists = [[], [], [], []]

    for fn in file_list:
        means, stds = get_results(args, fn)

        assert len(eval_lists) == len(means)
        for eval_list, mean in zip(eval_lists, means):
            eval_list.append(mean)

        assert len(std_lists) == len(stds)
        for std_list, std in zip(std_lists, stds):
            std_list.append(std)

    acc_mean = eval_lists[1:]
    acc_std = std_lists[1:]
    legends = legends[1:]
    draw(args, acc_mean, acc_std, legends, output_list[0], colors, lw, loc,
         labels, 'Accuracy (%)', 'Shared-Individual Layer Depths')


def get_steps(args, path):
    if args.first_experiment:
        exp_ids = ['1']
    else:
        exp_ids = ['1', '2', '3', '4', '5']
    results = [[], [], [], []]
    if args.plot_size < 0:
        length = -1
    else:
        length = args.plot_size + 1
    steps = None
    for e in exp_ids:
        fn = os.path.join(path + e, "log.txt")
        eval_list, steps = load(fn)
        assert len(results) == len(eval_list)
        for result, evaluation in zip(results, eval_list):
            result.append((evaluation[:length]))
    steps = steps[:length]
    for i in range(len(steps)):
        if i % 10 != 0:
            steps[i] = ''

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

    acc_mean = [eval3, eval4]
    acc_std = [std3, std4]
    draw(args, acc_mean, acc_std, legends, output_list[0], colors, lw, loc,
         steps, 'Accuracy (%)', 'Training Steps')


def main(args):
    font = {'family': 'serif'}
    rc('font', **font)

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
    parser.add_argument('--plot_size', type=int, default=-1,
                        help='Number of horizontal points to plot.')
    parser.add_argument('--first_experiment', action='store_true',
                        default=False,
                        help='Visualize first experiment.')
    parser.add_argument('--show_legend', action='store_true',
                        default=False, help='Show legend.')
    main(parser.parse_args())
