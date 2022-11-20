import os
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import rc


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
        ax.legend(loc=loc, prop={'size': 15})

    plt.savefig(basedir + '.pdf', bbox_inches='tight', pad_inches=0.01)


def write_draw(args, lists, stds, legends, basedir, colors, lw, loc, labels,
               v_name, u_name, plot=True, font_size=24):
    write_to_file(lists, stds, basedir)

    if plot:
        draw_figure(args, lists, stds, legends, basedir, colors, lw, loc,
                    labels, v_name, u_name, font_size=font_size)


def load(fn):
    date_format = '%a %b %d %H:%M:%S %Y'
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
        start_time = datetime.strptime(lines[0].strip().replace('CST ', ''),
                                       date_format)
        end_time = datetime.strptime(lines[1].strip().replace('CST ', ''),
                                     date_format)
        delta = end_time - start_time
        minutes = delta.total_seconds() / 60
    else:
        raise ValueError()
    return minutes


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
    loc = 'upper right'

    pairs = [
        ('Time.', ('black', 'D')),
    ]
    out_name = eid + 'time'

    legends = [x[0] for x in pairs]
    colors = [x[1] for x in pairs]
    output_name = os.path.join('outputs', args.experiment_id, out_name)

    return file_list, legends, output_name, colors, lw, loc, labels


def get_results(args, path):
    if args.first_experiment:
        exp_ids = ['1']
    else:
        exp_ids = ['1', '2', '3', '4', '5']
    results = []
    for e in exp_ids:
        fn = os.path.join(path + e, "time.txt")
        minutes = load(fn)
        results.append(minutes)

    matrix = np.asarray(results)
    means = np.mean(matrix)
    stds = np.std(matrix)

    return [means], [stds]


def main(args):
    font = {'family': 'serif'}
    rc('font', **font)

    file_list, legends, output_name, colors, lw, loc, labels = get_params(args)
    eval_lists = [[]]
    std_lists = [[]]

    for fn in file_list:
        means, stds = get_results(args, fn)

        assert len(eval_lists) == len(means)
        for eval_list, mean in zip(eval_lists, means):
            eval_list.append(mean)

        assert len(std_lists) == len(stds)
        for std_list, std in zip(std_lists, stds):
            std_list.append(std)

    write_draw(args, eval_lists, std_lists, legends, output_name, colors, lw,
               loc, labels, 'Time (mins)', 'Shared-Individual Network Depths')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', type=str,
                        default='review_dnn',
                        help='Experiment type.')
    parser.add_argument('--plot_size', type=int, default=-1,
                        help='Number of horizontal points to plot.')
    parser.add_argument('--first_experiment', action='store_true',
                        default=False,
                        help='Visualize first experiment.')
    parser.add_argument('--show_legend', action='store_true',
                        default=False, help='Show legend.')
    main(parser.parse_args())
