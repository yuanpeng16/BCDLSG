import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def draw_figure(args, lists, stds, basedir, labels, font_size=24):
    bottom = 90
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
    ax.set_ylabel('Angle', fontsize=font_size)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 5

    plt.savefig(basedir + '.pdf', bbox_inches='tight', pad_inches=0.01)


def main(args):
    font = {'family': 'serif'}
    rc('font', **font)

    folder = os.path.join("lin_results", str(args.width))
    os.makedirs(folder, exist_ok=True)
    fn = os.path.join(folder, 'results.txt')
    with open(fn, 'r') as f:
        lines = f.readlines()
    angles = []
    for line in lines:
        terms = line.strip().split()
        angles.append([float(term) for term in terms])
    angles = np.asarray(angles)
    matrix = np.transpose(angles)
    mean, std = matrix[1], matrix[2]
    labels = [str(2 ** i) for i in range(len(mean))]
    pdf_fn = os.path.join(folder, 'results')

    start = 3
    mean = mean[start:]
    std = std[start:]
    labels = labels[start:]
    draw_figure(args, mean, std, pdf_fn, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=32,
                        help='width.')
    main(parser.parse_args())
