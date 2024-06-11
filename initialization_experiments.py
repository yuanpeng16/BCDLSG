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

import jax.numpy as jnp
from jax import random
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Tanh, Relu, Softplus, Sigmoid, \
    Elu, LeakyRelu, Selu, Gelu
from jax.scipy.special import logsumexp
import keras.datasets.mnist


def get_activation(name):
    if name == 'tanh':
        return Tanh
    elif name == 'relu':
        return Relu
    elif name == 'softplus':
        return Softplus
    elif name == 'sigmoid':
        return Sigmoid
    elif name == 'elu':
        return Elu
    elif name == 'leakyrelu':
        return LeakyRelu
    elif name == 'selu':
        return Selu
    elif name == 'gelu':
        return Gelu
    elif name == 'linear':
        return None
    else:
        assert False


class Experiment(object):
    def __init__(self, args):
        self.depth = args.depth
        self.n_outputs = 10

        activation = get_activation(args.activation)
        layers = []
        for _ in range(self.depth):
            layers.append(Dense(args.width))
            if activation is not None:
                layers.append(activation)
        layers.append(Dense(self.n_outputs))
        self.init_random_params, self.logit_predict = stax.serial(*layers)

    def accuracy(self, params, inputs):
        predictions = self.predict(params, inputs)
        predicted_class = jnp.argmax(predictions, axis=1)
        size = len(predicted_class)
        predicted_class = np.eye(self.n_outputs)[predicted_class]
        freq = np.sum(predicted_class, axis=0)
        entropy = 0
        for f in freq:
            if f > 0:
                p = f / size
                entropy += - p * np.log2(p)
        return entropy

    def predict(self, params, inputs):
        logits = self.logit_predict(params, inputs)
        predictions = logits - logsumexp(logits, axis=1, keepdims=True)
        return predictions


def experiment(args):
    (train_images, train_labels), (
        test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = np.reshape(train_images, [train_images.shape[0], -1])
    test_images = np.reshape(test_images, [test_images.shape[0], -1])

    train_images = train_images / 256 - 0.5
    test_images = test_images / 256 - 0.5
    random_images = np.random.rand(*test_images.shape) - 0.5

    train_acc_list = []
    test_acc_list = []
    random_acc_list = []
    for _ in range(args.repeats):
        experiment = Experiment(args)
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


def main(args):
    lists = experiment(args)
    names = ["Train", "Test", "Random"]
    for x, name in zip(lists, names):
        x = np.asarray(x)
        mean = np.mean(x)
        std = np.std(x)
        y = np.power(2, x)
        y_mean = np.mean(y)
        y_std = np.std(y)
        print(name, '\t', mean, std, y_mean, y_std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=100,
                        help='Repeats.')
    parser.add_argument('--depth', type=int, default=8,
                        help='Depth.')
    parser.add_argument('--width', type=int, default=128,
                        help='Depth.')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation name.')
    main(parser.parse_args())
