import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def draw_figure(args, lists, stds, basedir, labels, font_size=24):
    bottom = np.min(lists - stds) - 0.05
    x_lim = len(lists)
    plt.figure(figsize=(9, 6))
    ax = plt.subplot(1, 1, 1)
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    l1 = lists - bottom
    ax.bar(np.arange(x_lim), l1, yerr=stds, align='center', alpha=0.5,
           color='black', ecolor='black', capsize=10, bottom=bottom)

    ax.set_xticks(range(x_lim))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Depth', fontsize=font_size)
    ax.set_ylabel('Rate', fontsize=font_size)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 5

    plt.savefig(basedir + '.pdf', bbox_inches='tight', pad_inches=0.01)


def main(args):
    font = {'family': 'serif'}
    rc('font', **font)

    matrix = []
    folder = os.path.join("same_outputs")
    for i in range(1, 9):
        fn = os.path.join(folder, "random_same_" + str(i) + '.txt')
        with open(fn, 'r') as f:
            lines = f.readlines()
        results = [float(line.strip().split()[1]) for line in lines]
        matrix.append(results)
    matrix = np.asarray(matrix)
    mean = np.mean(matrix, 1)
    std = np.std(matrix, 1)

    labels = [i for i in range(1, len(mean) + 1)]
    pdf_fn = os.path.join(folder, 'same_outputs_results')
    draw_figure(args, mean, std, pdf_fn, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=32,
                        help='width.')
    main(parser.parse_args())
