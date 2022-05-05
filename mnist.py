from __future__ import absolute_import, division, print_function, \
    unicode_literals

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
import argparse
import random
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
        x_list, y_list, y2_list = [], [], []
        for _ in range(k):
            y = random.randint(0, self.output_nodes - 1)
            x = random.choice(samples[y])
            y2 = self._get_y(y, isTrain)
            x = self._merge(x, y2, samples)
            x_list.append(x)
            y_list.append(self._one_hot(y))
            y2_list.append(self._one_hot(y2))
        x_list = np.asarray(x_list)
        y_list = np.asarray(y_list)
        y2_list = np.asarray(y2_list)
        return x_list, [y_list, y2_list]

    def get_training_samples(self, k):
        return self._get_samples(self.train_samples, k, isTrain=True)

    def get_eval_samples(self, k):
        return self.get_training_samples(k)

    def get_test_samples(self, k):
        return self._get_samples(self.test_samples, k, isTrain=False)


class LongDataGenerator(RandomDataGenerator):
    def _merge(self, x1, y2, labels):
        x = [0 * x1] * self.output_nodes
        x[y2] = x1

        x = np.concatenate(x, axis=1)
        x = np.expand_dims(x, -1)
        return x

    def get_input_shape(self):
        return (28, 28 * self.output_nodes, 1)


class PairedDataGenerator(RandomDataGenerator):
    def _merge(self, x1, y2, labels):
        x2 = random.choice(labels[y2])

        x = np.concatenate((x1, x2), axis=1)
        x = np.expand_dims(x, -1)
        return x

    def get_input_shape(self):
        return (28, 28 * 2, 1)


class StackedDataGenerator(RandomDataGenerator):
    def _merge(self, x1, y2, labels):
        x2 = random.choice(labels[y2])

        x1 = np.expand_dims(x1, -1)
        x2 = np.expand_dims(x2, -1)
        x = np.concatenate((x1, x2), axis=-1)
        return x

    def get_input_shape(self):
        return (28, 28, 2)


class DeepModelGenerator(object):
    def __init__(self, args, input_shape):
        self.args = args
        self.input_shape = input_shape

    def get_structure(self, output_nodes=10):
        inputs = Input(shape=self.input_shape)
        x = inputs
        for _ in range(2):
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        for _ in range(8):
            x = tf.keras.layers.Dense(64, activation='relu')(x)

        if self.args.loss_type == 'hinge':
            activation = 'linear'
        else:
            activation = 'softmax'
        outputs = [
            Dense(output_nodes, activation=activation, name='y1')(x),
            Dense(output_nodes, activation=activation, name='y2')(x)
        ]
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
    def __init__(self, args, model, datasets):
        self.args = args
        self.model = model
        self.datasets = datasets

    def evaluate(self, x, y):
        n_samples = len(y[0])
        y_hat = self.model.predict(x)
        hit1, hit2, hit = 0, 0, 0
        for i in range(n_samples):
            h1 = np.argmax(y[0][i]) == np.argmax(y_hat[0][i])
            h2 = np.argmax(y[1][i]) == np.argmax(y_hat[1][i])
            if h1:
                hit1 += 1
            if h2:
                hit2 += 1
            if h1 and h2:
                hit += 1
        acc = hit / n_samples
        acc1 = hit1 / n_samples
        acc2 = hit2 / n_samples
        return acc1, acc2, acc

    def evaluate_all(self):
        ret = []
        for data in self.datasets:
            ret.extend(self.evaluate(data[0], data[1]))
            ret.append("\t")
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
    elif args.merge_type == 'stacked':
        dg = StackedDataGenerator(args)
    else:
        assert False

    eval_data = dg.get_eval_samples(100)
    test_data = dg.get_test_samples(100)

    if args.save_image:
        for i in range(5):
            save_image(eval_data[0][i], 'eval_' + str(i) + '.png')
            save_image(test_data[0][i], 'test_' + str(i) + '.png')
        return

    mg = DeepModelGenerator(args, dg.get_input_shape())
    model = mg.get_model()
    ev = Evaluator(args, model, [eval_data, test_data])

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
    parser.add_argument('--mask_input', type=int, default=0,
                        help='mask a particular input.')
    parser.add_argument('--n_hidden_layers', type=int, default=2,
                        help='Number of hidden layer.')
    parser.add_argument('--n_hidden_nodes', type=int, default=32,
                        help='Number of nodes in hidden layer.')
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
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
    args = parser.parse_args()
    main(args)
