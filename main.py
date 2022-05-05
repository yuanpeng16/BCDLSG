# This code is compatible with Tensorflow v1

import argparse
import random
import numpy as np

from data_generator import Formarter
from data_generator import get_data_generator
from s2s_model import S2SModel


def random_initialize(seed):
    random.seed(seed)
    if args.random_random:
        np.random.seed(random.randint(2, 1000))
    else:
        np.random.seed(seed)


def choose(train_X, train_Y, train_X_len, train_Y_len):
    pop = list(range(len(train_X)))
    indice = random.choices(pop, k=100)
    x, y, x_len, y_len = [], [], [], []
    for index in indice:
        x.append(train_X[index])
        y.append(train_Y[index])
        x_len.append(train_X_len[index])
        y_len.append(train_Y_len[index])
    return x, y, x_len, y_len


def main(args):
    # prepare data
    random_initialize(43)
    dg = get_data_generator(args.data_name, args)
    train_X, train_Y = dg.get_train_data()
    test_X, test_Y = dg.get_test_data()

    random_initialize(args.random_seed)

    if args.use_start_symbol:
        train_X = [['S'] + x for x in train_X]
        test_X = [['S'] + x for x in test_X]

    # change format
    formater = Formarter(args)
    samples, dicts, lengths, maxs = formater.initialize(
        train_X, train_Y, test_X, test_Y)
    train_X, train_Y, test_X, test_Y = samples
    voc, act = dicts
    train_X_len, train_Y_len, test_X_len, test_Y_len = lengths

    if args.remove_x_eos:
        train_X_len = [x - 1 for x in train_X_len]
        test_X_len = [x - 1 for x in test_X_len]

    max_input, max_output = maxs

    args.input_length = max_input
    args.output_length = max_output

    # prepare model
    model = S2SModel(args)
    model.initialize(len(voc) + 1, len(act) + 1)
    train_data = train_X, train_Y, train_X_len, train_Y_len
    val_data = choose(*train_data)
    test_data = choose(test_X, test_Y, test_X_len, test_Y_len)
    model.train(train_data, val_data, test_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', type=str, default='default',
                        help='experiment ID')
    parser.add_argument('--model_name', type=str, default='normal',
                        help='model name')
    parser.add_argument('--print_output', action='store_true', default=False,
                        help='Linear max.')
    parser.add_argument('--simple_data', action='store_true', default=False,
                        help='use simple data.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--learning_rate', type=float, default=0.3,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch_size')
    parser.add_argument('--shuffle_batch', action='store_true', default=False,
                        help='shuffle batch.')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='epochs')
    parser.add_argument('--data_name', type=str, default='scan',
                        help='name of data set')
    parser.add_argument('--train_file', type=str,
                        default='SCAN/add_prim_split/tasks_train_addprim_jump.txt',
                        help='train file name')
    parser.add_argument('--test_file', type=str,
                        default='SCAN/add_prim_split/tasks_test_addprim_jump.txt',
                        help='test file name')
    parser.add_argument('--switch_temperature', type=float, default=1.0,
                        help='switch temperature')
    parser.add_argument('--attention_temperature', type=float, default=10.0,
                        help='attention temperature')
    parser.add_argument('--num_units', type=int, default=16,
                        help='num units')
    parser.add_argument('--bidirectional_encoder', action='store_true',
                        default=False,
                        help='bidirectional encoder.')
    parser.add_argument('--max_gradient_norm', type=float, default=-1.0,
                        help='max gradient norm')
    parser.add_argument('--decay_steps', type=int, default=-1,
                        help='decay steps')
    parser.add_argument('--use_embedding', action='store_true', default=False,
                        help='use embedding.')
    parser.add_argument('--embedding_size', type=int, default=32,
                        help='embedding size')
    parser.add_argument('--reg_coe', type=float, default=-1.0,
                        help='regularization coeficient')
    parser.add_argument('--use_start_symbol', action='store_true',
                        default=False,
                        help='use start symbol')
    parser.add_argument('--content_noise', action='store_true', default=False,
                        help='add noise to content')
    parser.add_argument('--content_noise_coe', type=float, default=-1.0,
                        help='noise regularization coeficient')
    parser.add_argument('--sample_wise_content_noise', action='store_true',
                        default=False,
                        help='sample-wise noise regularization')
    parser.add_argument('--function_noise', action='store_true', default=False,
                        help='add noise to function')
    parser.add_argument('--remove_x_eos', action='store_true', default=False,
                        help='remove x eos')
    parser.add_argument('--masked_attention', action='store_true',
                        default=False,
                        help='masked attention')
    parser.add_argument('--use_entropy_reg', action='store_true',
                        default=False,
                        help='use entropy reg')
    parser.add_argument('--random_random', action='store_true', default=False,
                        help='random_random')
    parser.add_argument('--single_representation', action='store_true',
                        default=False,
                        help='single representation')
    parser.add_argument('--use_decoder_input', action='store_true',
                        default=False,
                        help='single representation')
    parser.add_argument('--output_embedding_size', type=int, default=8,
                        help='output embedding size')
    parser.add_argument('--use_l1_norm', action='store_true', default=False,
                        help='single representation')
    parser.add_argument('--remove_prediction_bias', action='store_true',
                        default=False,
                        help='remove prediction bias')
    parser.add_argument('--clip_by_norm', action='store_true', default=False,
                        help='clip by norm instead of global norm.')
    parser.add_argument('--mask_jump', action='store_true', default=False,
                        help='Mask jamp.')
    parser.add_argument('--hinge_loss', action='store_true', default=False,
                        help='Use hinge loss.')
    args = parser.parse_args()

    main(args)
