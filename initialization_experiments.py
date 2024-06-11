# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic MNIST example using JAX with the mini-libraries stax and optimizers.

The mini-library jax.example_libraries.stax is for neural network building, and
the mini-library jax.example_libraries.optimizers is for first-order stochastic
optimization.
"""

import numpy as np
import numpy.random as npr
import argparse

from jax import random
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Tanh, Relu, Softplus, Sigmoid, \
    Elu, LeakyRelu, Selu, Gelu
from jax.scipy.special import logsumexp
import keras.datasets.mnist


class Experiment(object):
    def __init__(self, args, activation):
        self.depth = args.depth
        self.n_outputs = 10

        layers = []
        for _ in range(self.depth):
            layers.append(Dense(args.width))
            if activation is not None:
                layers.append(activation)
        layers.append(Dense(self.n_outputs))
        self.init_random_params, self.logit_predict = stax.serial(*layers)

    def accuracy(self, params, inputs):
        log_p = self.predict(params, inputs)
        p = np.exp(log_p)
        denominator = np.sum(p, axis=1, keepdims=True)
        p = p / denominator
        log_p = np.log2(p)

        average_p = np.mean(p, axis=0, keepdims=True)
        cross_entropy = -np.sum(average_p * log_p, axis=1)
        average_cross_entropy = np.mean(cross_entropy)
        average_entropy = -np.sum(average_p * np.log2(average_p))
        kl_divergence = average_cross_entropy - average_entropy
        return kl_divergence

    def predict(self, params, inputs):
        logits = self.logit_predict(params, inputs)
        predictions = logits - logsumexp(logits, axis=1, keepdims=True)
        return predictions


def one_experiment(args, activation, images):
    train_images, test_images, random_images = images
    train_acc_list = []
    test_acc_list = []
    random_acc_list = []
    for _ in range(args.repeats):
        experiment = Experiment(args, activation)
        rng = random.PRNGKey(npr.randint(1000000))
        _, init_params = experiment.init_random_params(rng, (-1, 28 * 28))
        train_acc = experiment.accuracy(init_params, train_images)
        test_acc = experiment.accuracy(init_params, test_images)
        random_acc = experiment.accuracy(init_params, random_images)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        random_acc_list.append(random_acc)
    lists = train_acc_list, test_acc_list, random_acc_list
    return lists


def one_type(args, activation, images):
    lists = one_experiment(args, activation, images)
    names = ["Train", "Test", "Random"]
    for x, name in zip(lists, names):
        x = np.asarray(x)
        mean = np.mean(x)
        std = np.std(x)
        y = np.power(2, x)
        y_mean = np.mean(y)
        y_std = np.std(y)
        print(name, '\t', mean, std, y_mean, y_std)


def main(args):
    types = [
        ['Tanh', Tanh],
        ['Relu', Relu],
        ['Softplus', Softplus],
        ['Sigmoid', Sigmoid],
        ['Elu', Elu],
        ['LeakyRelu', LeakyRelu],
        ['Selu', Selu],
        ['Gelu', Gelu],
        ['Linear', None]
    ]
    if args.activation is not None:
        m = {x: y for x, y in types}
        assert args.activation in m
        types = [[args.activation, m[args.activation]]]

    (train_images, train_labels), (
        test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = np.reshape(train_images, [train_images.shape[0], -1])
    test_images = np.reshape(test_images, [test_images.shape[0], -1])

    train_images = train_images / 256 - 0.5
    test_images = test_images / 256 - 0.5
    random_images = np.random.rand(*test_images.shape) - 0.5
    images = train_images, test_images, random_images

    for name, activation in types:
        print(name)
        one_type(args, activation, images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=100,
                        help='Repeats.')
    parser.add_argument('--depth', type=int, default=8,
                        help='Depth.')
    parser.add_argument('--width', type=int, default=128,
                        help='Depth.')
    parser.add_argument('--activation', type=str, default=None,
                        help='Activation name.')
    main(parser.parse_args())
