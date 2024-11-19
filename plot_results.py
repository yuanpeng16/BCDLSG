import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
# from brokenaxes import brokenaxes


def write_to_file(lists, stds, basedir):
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


def draw_figure_broken(args, lists, stds, legends, basedir, colors, lw, loc,
                       labels, v_name, u_name, font_size=24):
    x_lim = len(lists[0])
    plt.figure(figsize=(9, 6))

    ls = np.asarray(lists)
    ss = np.asarray(stds)
    lower = np.min(ls[:2] - ss[:2]) - 5
    upper = np.max(ls[2:] + ss[2:]) + 5
    if upper > lower:
        ylims = ((-1, 45), (85, 100))
    else:
        ylims = ((-1, upper), (lower, 100))

    # ax = brokenaxes(subplot_spec=None, ylims=ylims, hspace=.05)
    ax = None

    ax.tick_params(axis='both', which='major', labelsize=font_size)

    for i, (l, s, legend, entries) in enumerate(
            zip(lists, stds, legends, colors)):
        color, marker = entries
        l1 = np.asarray(l)
        s1 = np.asarray(s)
        ls = '-' if i % 2 == 0 else '--'
        ax.plot(l1, lw=lw, markevery=(0, 1), ls=ls, marker=marker,
                markersize=16, markeredgewidth=2, markerfacecolor='none',
                color=color, label=legend)
        ax.fill_between(np.arange(x_lim), l1 - s1, l1 + s1, color=color,
                        alpha=0.2)

    ax.set_xlim([0, x_lim - 1])
    ax.set_xticklabels([''] + labels)
    ax.set_xticks(range(x_lim))
    # ax.set_xlabel(u_name, fontsize=font_size)
    # ax.set_ylabel(v_name, fontsize=font_size)
    # ax.xaxis.labelpad = 5
    # ax.yaxis.labelpad = 5

    if args.show_legend:
        if args.legend_font_size < 0:
            legend_font = font_size
        else:
            legend_font = args.legend_font_size
        ax.legend(loc=loc, prop={'size': legend_font})

    plt.savefig(basedir + '.pdf', bbox_inches='tight', pad_inches=0.01)


def draw_figure(args, lists, stds, legends, basedir, colors, lw, loc, labels,
                v_name, u_name, font_size=24):
    x_lim = len(lists[0])
    plt.figure(figsize=(9, 6))
    ax = plt.subplot(1, 1, 1)
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    for i, (l, s, legend, entries) in enumerate(
            zip(lists, stds, legends, colors)):
        color, marker = entries
        l1 = np.asarray(l)
        s1 = np.asarray(s)
        ls = '-' if i % 2 == 0 else '--'
        ax.plot(l1, lw=lw, markevery=(0, 1), ls=ls, marker=marker,
                markersize=16, markeredgewidth=2, markerfacecolor='none',
                color=color, label=legend)
        ax.fill_between(np.arange(x_lim), l1 - s1, l1 + s1, color=color,
                        alpha=0.2)

    ax.set_xlim([0, x_lim - 1])
    ax.set_xticks(range(x_lim))
    ax.set_xticklabels(labels)
    ax.set_xlabel(u_name, fontsize=font_size)
    ax.set_ylabel(v_name, fontsize=font_size)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 5

    if args.show_legend:
        if args.legend_font_size < 0:
            legend_font = font_size
        else:
            legend_font = args.legend_font_size
        ax.legend(loc=loc, prop={'size': legend_font})

    plt.savefig(basedir + '.pdf', bbox_inches='tight', pad_inches=0.01)


def write_draw(args, lists, stds, legends, basedir, colors, lw, loc, labels,
               v_name, u_name, plot=True, font_size=24):
    write_to_file(lists, stds, basedir)

    if args.plot_type == "evaluation":
        lists = lists[2:]
        stds = stds[2:]
        legends = legends[2:]
        colors = colors[2:]
        basedir += "_eval"
    elif args.plot_type == "simple":
        lists = lists[2:3]
        stds = stds[2:3]
        legends = ["Accuracy (%) for\ncompositional generalization"]
        colors = colors[2:3]
        v_name = None
        u_name = "Degree of parameter sharing"
        labels = ["small"] + [None] * (len(labels) - 2) + ["large"]
        basedir += "_simple"
        font_size = 28
    if plot:
        draw_figure(args, lists, stds, legends, basedir, colors, lw, loc,
                    labels, v_name, u_name, font_size=font_size)


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
        lines = [x for x in lines if not x[0] == 'loading']
        index = [3, 18, 8, 9, 14]
        steps = get_list_index(lines)
        eval_list = [get_list(lines, i) for i in index]
    else:
        eval_list = [[], [], [], []]
    return eval_list, steps


def get_params(args):
    if args.experiment_id.endswith('_resnet'):
        depth = 5
    elif args.experiment_id.endswith('_lstm-1'):
        depth = 2
    else:
        depth = 7

    ids = [str(i) + '_' + str(depth - i) for i in range(depth + 1)]
    labels = [str(i) + '-' + str(depth - i) for i in range(depth + 1)]
    eid = args.experiment_id + '_'
    log_dir = os.path.join('logs', args.experiment_id)
    file_list = [os.path.join(log_dir, eid + c + '_') for c in ids]
    lw = 2
    loc = 'best'

    if args.experiment_type == 'main':
        pairs = [
            ('Train Sample Acc.', ('brown', 'D')),
            ('I.I.D. Sample Acc.', ('orange', 'D')),
            ('Test Sample Acc.', ('b', 'v')),
            ('Test Set Acc.', ('c', '^')),
            ('Rnd. Set Acc.', ('r', 's')),
        ]
        out_name = eid + 'acc'
    elif args.experiment_type == 'steps':
        pairs = [
            ('Test Set Acc.', ('c', '')),
            ('Rnd. Set Acc.', ('r', '')),
        ]
        out_name = eid + 'steps_acc'
    else:
        raise ValueError(
            '{0} is not a valid experiment_type.'.format(args.experiment_type))

    legends = [x[0] for x in pairs]
    colors = [x[1] for x in pairs]
    output_name = os.path.join('outputs', args.experiment_id, out_name)

    return file_list, legends, output_name, colors, lw, loc, labels


def get_results(args, path):
    if args.first_experiment:
        exp_ids = ['1']
    else:
        exp_ids = ['1', '2', '3', '4', '5']
    results = [[], [], [], [], []]
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


def final_main(args):
    file_list, legends, output_name, colors, lw, loc, labels = get_params(args)
    eval_lists = [[], [], [], [], []]
    std_lists = [[], [], [], [], []]

    for fn in file_list:
        means, stds = get_results(args, fn)

        assert len(eval_lists) == len(means)
        for eval_list, mean in zip(eval_lists, means):
            eval_list.append(mean)

        assert len(std_lists) == len(stds)
        for std_list, std in zip(std_lists, stds):
            std_list.append(std)

    acc_mean = eval_lists
    acc_std = std_lists
    write_draw(args, acc_mean, acc_std, legends, output_name, colors, lw,
               loc, labels, 'Accuracy (%)', 'Shared-Individual Network Depths')


def get_step_results(args, path):
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
    file_list, legends, output_name, colors, lw, loc, labels = get_params(args)

    fn = file_list[-1]
    means, stds, steps = get_step_results(args, fn)

    acc_mean = means[2:]
    acc_std = stds[2:]
    write_draw(args, acc_mean, acc_std, legends, output_name, colors, lw,
               loc, steps, 'Accuracy (%)', 'Training Steps')


def main(args):
    font = {'family': 'serif'}
    rc('font', **font)

    if args.experiment_type == 'main':
        final_main(args)
    elif args.experiment_type == 'steps':
        step_main(args)
    else:
        raise ValueError(
            '{0} is not a valid experiment_type.'.format(args.experiment_type))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_type', type=str, default='main',
                        help='Experiment type.')
    parser.add_argument('--experiment_id', type=str,
                        default='main_dnn',
                        help='Experiment type.')
    parser.add_argument('--plot_size', type=int, default=-1,
                        help='Number of horizontal points to plot.')
    parser.add_argument('--first_experiment', action='store_true',
                        default=False,
                        help='Visualize first experiment.')
    parser.add_argument('--show_legend', action='store_true',
                        default=False, help='Show legend.')
    parser.add_argument('--legend_font_size', type=int, default=15,
                        help='Number of horizontal points to plot.')
    parser.add_argument('--plot_type', type=str,
                        default='evaluation', help='Experiment type.')
    main(parser.parse_args())
