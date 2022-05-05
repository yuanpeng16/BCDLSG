from __future__ import absolute_import, division, print_function, \
    unicode_literals

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
import argparse
import random
import math
import numpy as np
from PIL import Image


class RandomDataGenerator(object):
    def __init__(self, args):
        self.args = args
        self.output_nodes = 10

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.train_samples = self._prepare_data(x_train, y_train)
        self.test_samples = self._prepare_data(x_test, y_test)

    def _prepare_data(self, x_all, y_all):
        assert len(x_all) == len(y_all)
        x_all = x_all / 255.0
        x_all = x_all.astype("float32")

        data = [[] for _ in range(self.output_nodes)]
        for x, y in zip(x_all, y_all):
            y = int(y)
            data[y].append(x)
        return data

    def _one_hot(self, a):
        ret = [0] * self.output_nodes
        ret[a] = 1
        return ret

    def _get_y(self, y1, isTrain):
        buf = 2
        if isTrain:
            if self.args.mask_input == 2:
                # second setting
                y = random.randint(0, self.output_nodes - 1)
            else:
                # first setting
                y = (random.randint(0, buf - 1) + y1)
        else:
            y = (random.randint(0, self.output_nodes - 1 - buf) + y1 + buf)
        y = y % self.output_nodes
        return y

    def _get_samples(self, samples, k, isTrain):
        x_list, y_list = [], []
        for _ in range(k):
            y = random.randint(0, self.output_nodes - 1)
            x = random.choice(samples[y])
            y2 = self._get_y(y, isTrain)
            x = self._merge(x, y2, samples)
            x_list.append(x)
            y_list.append(self._one_hot(y))
        x_list = np.asarray(x_list)
        y_list = np.asarray(y_list)
        return x_list, y_list

    def _get_mi_samples(self, samples, k, n):
        x_list, y_list = [], []
        for _ in range(k):
            y = random.randint(0, self.output_nodes - 1)
            x = random.choice(samples[y])
            mi_list = []
            for _ in range(n):
                y2 = random.randint(0, self.output_nodes - 1)
                mi_list.append(self._merge(x, y2, samples))
            x_list.append(mi_list)
            y_list.append(self._one_hot(y))
        x_list = np.asarray(x_list)
        y_list = np.asarray(y_list)
        return x_list, y_list

    def get_training_samples(self, k):
        return self._get_samples(self.train_samples, k, isTrain=True)

    def get_eval_samples(self, k):
        return self.get_training_samples(k)

    def get_test_samples(self, k):
        return self._get_samples(self.test_samples, k, isTrain=False)

    def get_eval_mi_samples(self, k, n):
        return self._get_mi_samples(self.train_samples, k, n)

    def get_test_mi_samples(self, k, n):
        return self._get_mi_samples(self.test_samples, k, n)


class LongDataGenerator(RandomDataGenerator):
    def _merge(self, x1, y2, labels):
        x = [0 * x1] * self.output_nodes
        x[y2] = x1

        x = np.concatenate(x, axis=1)
        return x

    def get_input_shape(self):
        return (28, 28 * self.output_nodes)


class PairedDataGenerator(RandomDataGenerator):
    def _merge(self, x1, y2, labels):
        x2 = random.choice(labels[y2])

        x = np.concatenate((x1, x2), axis=1)
        return x

    def get_input_shape(self):
        return (28, 28 * 2)


class DeepModelGenerator(object):
    def __init__(self, args, input_shape):
        self.args = args
        self.input_shape = input_shape

    def get_structure(self, output_nodes=10):
        inputs = Input(shape=self.input_shape)
        x = tf.expand_dims(inputs, -1)

        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)

        if self.args.loss_type == 'hinge':
            activation = 'linear'
        else:
            activation = 'softmax'
        outputs = Dense(output_nodes, activation=activation, name='y1')(x)
        if self.args.two_outputs:
            outputs = [outputs,
                       Dense(output_nodes, activation=activation, name='y2')(
                           x)]
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def get_model(self):
        model = self.get_structure()
        adam = tf.keras.optimizers.Adam(lr=self.args.lr)
        if self.args.loss_type == 'hinge':
            loss = 'categorical_hinge'
        else:
            loss = 'categorical_crossentropy'
        model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
        return model


class Evaluator(object):
    def __init__(self, args, model, datasets, mi_datasets):
        self.args = args
        self.model = model
        self.datasets = datasets
        self.mi_datasets = [self.preprocess(data[0]) for data in mi_datasets]

    def preprocess(self, data):
        x_list = []
        length_list = []
        for x in data:
            x_list.extend(x)
            length_list.append(len(x))
        x_list = np.asarray(x_list)
        return x_list, length_list

    def predict(self, x, y):
        y_hat = self.model.predict(x)
        for a, b in zip(y, y_hat):
            for c, d in zip(a, b):
                print(c, d)

    def predict_all(self):
        ret = []
        for data in self.datasets:
            ret.extend(self.predict(data[0], data[1]))
        return ret

    def evaluate(self, x, y):
        if self.args.two_outputs:
            loss, loss1, loss2, acc1, acc2 = self.model.evaluate(x, y,
                                                                 verbose=0)
            return loss, loss1, loss2, acc1, acc2
        else:
            loss, acc1 = self.model.evaluate(x, y, verbose=0)
            return loss, acc1

    def evaluate_all(self):
        ret = []
        for data in self.datasets:
            ret.extend(self.evaluate(data[0], data[1]))

        if self.args.compute_entropy:
            ret.extend(self.compute_mi_all())
        return ret

    def entropy(self, y_list):
        n = len(y_list)
        y_count = {}
        for y in y_list:
            y_count[y] = y_count.get(y, 0) + 1
        entropy = 0
        for _, v in y_count.items():
            p = v / n
            entropy -= p * math.log2(p)
        return entropy

    def compute_mi(self, data):
        x_list, length_list = data
        y_dist_list = self.model.predict(x_list)
        y_list_all = np.argmax(y_dist_list, axis=-1)
        avg_entropy = 0
        offset = 0
        for length in length_list:
            end = length + offset
            y_list = y_list_all[offset:end]
            avg_entropy += self.entropy(y_list)
            offset = end
        return avg_entropy / len(length_list)

    def compute_mi_all(self):
        ret = []
        for data in self.mi_datasets:
            ret.append(self.compute_mi(data))
        return ret


def get_grad_norm(loss_object, model, x_train, y_train):
    x_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        predictions = model(x_tensor)
        loss = loss_object(y_train, predictions)
    gradients = tape.gradient(loss, x_tensor)
    norm = tf.linalg.global_norm([gradients]).numpy()
    return norm


def save_image(x, path):
    Image.fromarray(255 * x).convert('RGB').save(path)


def main(args):
    # set random seeds
    random.seed(args.data_random_seed)
    tf.random.set_seed(args.parameter_random_seed)

    # get data
    if args.merge_type == 'slide':
        dg = LongDataGenerator(args)
    elif args.merge_type == 'paired':
        dg = PairedDataGenerator(args)
    else:
        assert False

    eval_data = dg.get_eval_samples(100)
    test_data = dg.get_test_samples(100)

    mi_data = []
    if args.compute_entropy:
        mi_data.append(dg.get_eval_mi_samples(100, 100))
        mi_data.append(dg.get_test_mi_samples(100, 100))

    if args.save_image:
        for i in range(5):
            save_image(eval_data[0][i], 'eval_' + str(i) + '.png')
            save_image(test_data[0][i], 'test_' + str(i) + '.png')
        return

    mg = DeepModelGenerator(args, dg.get_input_shape())
    model = mg.get_model()
    ev = Evaluator(args, model, [eval_data, test_data], mi_data)

    loss_object = tf.keras.losses.CategoricalHinge()

    # train and evaluate
    print(0, 0, *ev.evaluate_all())
    for i in range(args.steps):
        x_train, y_train = dg.get_training_samples(args.batch_size)

        if args.compute_gradient:
            norm = get_grad_norm(loss_object, model, x_train, y_train)
        else:
            norm = 0

        model.fit(x_train, y_train,
                  validation_data=test_data,
                  batch_size=args.batch_size,
                  epochs=1, verbose=0)
        if i % 1 == 0:
            print(i + 1, norm, *ev.evaluate_all())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_random_seed', type=int, default=8,
                        help='Random seed.')
    parser.add_argument('--parameter_random_seed', type=int, default=7,
                        help='Random seed.')
    parser.add_argument('--two_outputs', action='store_true', default=False,
                        help='Use two outputs')
    parser.add_argument('--mask_input', type=int, default=0,
                        help='mask a particular input.')
    parser.add_argument('--n_hidden_layers', type=int, default=2,
                        help='Number of hidden layer.')
    parser.add_argument('--n_hidden_nodes', type=int, default=32,
                        help='Number of nodes in hidden layer.')
    parser.add_argument('--loss_type', type=str, default='hinge',
                        help='Loss type.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--steps', type=int, default=500,
                        help='Steps.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--merge_type', type=str, default='paired',
                        help='Merge type.')
    parser.add_argument('--random_threshold', type=float, default=0.85,
                        help='Threshold to randomize the second input.')
    parser.add_argument('--save_image', action='store_true', default=False,
                        help='Show image and stop.')
    parser.add_argument('--compute_gradient', action='store_true',
                        default=False, help='Compute gradient.')
    parser.add_argument('--compute_entropy', action='store_true',
                        default=False, help='Compute entropy.')
    args = parser.parse_args()
    main(args)
