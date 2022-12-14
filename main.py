from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import tensorflow as tf
import argparse
import random
import os
from PIL import Image

from data import get_data_generator
from model import get_model_generator
from evaluator import get_evaluator


def set_random_seeds(parameter_random_seed, data_random_seed):
    random.seed(data_random_seed)
    np.random.seed(data_random_seed)
    tf.random.set_seed(parameter_random_seed)


def save_image(x, path):
    x = np.squeeze(x)
    x = np.minimum(np.maximum(x + 0.5, 0), 1)
    Image.fromarray(np.uint8(255 * x)).convert('RGB').save(path)


def train(args, dg, model, ev, pretrain=False):
    for i in range(args.steps):
        x_train, y_train = dg.get_training_samples(args.batch_size,
                                                   pretrain=pretrain)
        model.fit(x_train, y_train, batch_size=args.batch_size, epochs=1,
                  verbose=0)
        if i % args.log_interval == args.log_interval - 1:
            print(i + 1, *ev.evaluate_all())


def main(args):
    # get data (fix test data)
    set_random_seeds(42, 43)
    dg = get_data_generator(args)
    eval_data = dg.get_eval_samples(args.test_sample_size)
    test_data = dg.get_test_samples(args.test_sample_size, randomize=False)
    random_data = dg.get_test_samples(args.test_sample_size, randomize=True)
    large_eval_data = dg.get_eval_samples(10 * args.test_sample_size)
    large_test_data = dg.get_test_samples(10 * args.test_sample_size,
                                          randomize=False)
    large_random_data = dg.get_test_samples(10 * args.test_sample_size,
                                            randomize=True)

    # set random seeds
    set_random_seeds(args.parameter_random_seed, args.data_random_seed)

    if args.save_image:
        for i in range(5):
            save_image(large_eval_data[0][i],
                       os.path.join(args.log_dir, 'eval_' + str(i) + '.png'))
            save_image(large_test_data[0][i],
                       os.path.join(args.log_dir, 'test_' + str(i) + '.png'))
            save_image(large_random_data[0][i],
                       os.path.join(args.log_dir, 'random_' + str(i) + '.png'))

    # get model and evaluator
    output_nodes = dg.get_output_nodes()
    mg = get_model_generator(args, dg.get_input_shape(), output_nodes)
    mg.set_vocab_size(dg.get_vocab_size())
    model = mg.get_model()
    test_label_pairs = dg.get_test_label_pairs()
    ev = get_evaluator(args, model, [eval_data, test_data, random_data],
                       [large_eval_data, large_test_data, large_random_data],
                       test_label_pairs)

    # train and evaluate
    print(0, *ev.evaluate_all())
    if args.pretrain:
        train(args, dg, model, ev, pretrain=True)
        print("pretrain", *ev.large_evaluate_all())
    train(args, dg, model, ev)
    print("final", *ev.large_evaluate_all())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='',
                        help='Log directory.')
    parser.add_argument('--data_random_seed', type=int, default=8,
                        help='Random seed.')
    parser.add_argument('--parameter_random_seed', type=int, default=7,
                        help='Random seed.')
    parser.add_argument('--n_shared_layers', type=int, default=2,
                        help='Number of shared layer.')
    parser.add_argument('--n_individual_layers', type=int, default=1,
                        help='Number of individual layer.')
    parser.add_argument('--n_hidden_nodes', type=int, default=32,
                        help='Number of nodes in hidden layer.')
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                        help='Loss type.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--test_sample_size', type=int, default=1000,
                        help='Test sample size.')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval.')
    parser.add_argument('--steps', type=int, default=500,
                        help='Steps.')
    parser.add_argument('--combined_labels', type=int, default=3,
                        help='Combined labels.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--merge_type', type=str, default='paired',
                        help='Merge type.')
    parser.add_argument('--evaluator_type', type=str, default='normal',
                        help='Evaluator type.')
    parser.add_argument('--save_image', action='store_true', default=False,
                        help='Show image and stop.')
    parser.add_argument('--adversarial', action='store_true',
                        default=False,
                        help='Use adversarial learning on test.')
    parser.add_argument('--dataset1', type=str, default='mnist',
                        help='Test Dataset.')
    parser.add_argument('--dataset2', type=str, default='mnist',
                        help='Test Dataset.')
    parser.add_argument('--any_generalization', action='store_true',
                        default=False, help='Any systematic generalization.')
    parser.add_argument('--model_type', type=str, default='cnn',
                        help='Model type.')
    parser.add_argument('--input_permutation', action='store_true',
                        default=False, help='Permute input.')
    parser.add_argument('--label_split', type=str, default='tile',
                        help='Model type.')
    parser.add_argument('--rotate_second_input', action='store_true',
                        default=False, help='Rotate second input.')
    parser.add_argument('--pretrain', action='store_true',
                        default=False, help='Rotate second input.')
    parser.add_argument('--dataset_dir', type=str,
                        default='../../data/zeroshot_datasets',
                        help='Zero-shot dataset directory.')
    parser.add_argument('--partition_threshold_percentage', type=int,
                        default=50,
                        help='Partition threshold percentage for evaluation.')
    main(parser.parse_args())
