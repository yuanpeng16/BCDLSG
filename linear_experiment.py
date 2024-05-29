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

import itertools

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import os
import argparse

import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu
from jax.scipy.special import logsumexp


# from examples import datasets

def get_labels(inputs):
    return np.maximum(np.sign(inputs), 0).astype(np.int32)


def get_data():
    train_inputs = np.asarray([
        [-1, 1],
        [-1, -1],
        [1, -1]
    ], dtype=np.float32)
    train_labels = get_labels(train_inputs)

    test_inputs = []
    dots = 50
    for i in range(-dots, dots, 1):
        for j in range(-dots, dots, 1):
            test_inputs.append([i / dots, j / dots])
    test_inputs = np.asarray(test_inputs, dtype=np.float32)
    test_labels = np.asarray([[1, 1] for _ in test_inputs])
    return (train_inputs, train_labels), (test_inputs, test_labels)


class Experiment(object):
    def __init__(self, args, depth):
        self.args = args
        self.depth = depth
        layers = []
        for _ in range(depth):
            layers += [Dense(args.width)]
        layers.append(Dense(4))
        self.init_random_params, self.logit_predict = stax.serial(*layers)

    def save_fig(self, inputs, predicted_class, fn):
        predicted_class1, predicted_class2 = predicted_class
        prediction = [[] for _ in range(4)]
        for x, p1, p2 in zip(inputs, predicted_class1, predicted_class2):
            prediction[p1 * 2 + p2].append(x)

        colors = ['red', 'green', 'blue', 'yellow']
        for p, c in zip(prediction, colors):
            if len(p) > 0:
                x, y = np.transpose(np.asarray(p))
                plt.scatter(x, y, c=c)
        plt.savefig(fn)
        plt.clf()

    def loss(self, params, batch):
        inputs, targets = batch
        targets1, targets2 = targets
        preds1, preds2 = self.predict(params, inputs)
        loss1 = -jnp.mean(jnp.sum(preds1 * targets1, axis=1))
        loss2 = -jnp.mean(jnp.sum(preds2 * targets2, axis=1))
        return loss1 + loss2

    def accuracy1(self, params, batch, save=None):
        inputs, targets = batch
        targets1, targets2 = targets
        target_class1 = jnp.argmax(targets1, axis=1)
        target_class2 = jnp.argmax(targets2, axis=1)
        preds1, preds2 = self.predict(params, inputs)
        predicted_class1 = jnp.argmax(preds1, axis=1)
        predicted_class2 = jnp.argmax(preds2, axis=1)
        if save is not None:
            predicted_class = predicted_class1, predicted_class2
            self.save_fig(inputs, predicted_class, save)
        correct1 = predicted_class1 == target_class1
        correct2 = predicted_class2 == target_class2
        return jnp.mean(correct1 * correct2)

    def get_angle(self, params):
        w, _ = params[0]
        for w1, _ in params[1:]:
            w = np.matmul(w, w1)
        w = np.transpose(w)
        v1 = w[0] - w[1]
        v2 = w[2] - w[3]
        v1_length = np.linalg.norm(v1)
        v2_length = np.linalg.norm(v2)
        cos = np.dot(v1, v2) / (v1_length * v2_length)
        assert cos > -1 - 0.000001
        assert cos < 1 + 0.000001
        angle = np.arccos(np.clip(cos, -1, 1))
        angle = np.degrees(angle)
        return angle

    def get_linear_params(self, params):
        w, b = params[0]
        for w1, b1 in params[1:]:
            w = np.matmul(w, w1)
            b = np.matmul(b, w1) + b1
        return w, b

    def accuracy(self, params, batch, save=None):
        inputs, targets = batch
        targets1, targets2 = targets
        target_class1 = jnp.argmax(targets1, axis=1)
        target_class2 = jnp.argmax(targets2, axis=1)

        w, b = self.get_linear_params(params)
        predictions = np.matmul(inputs, w) + np.expand_dims(b, 0)
        preds1, preds2 = np.split(predictions, 2, -1)

        predicted_class1 = jnp.argmax(preds1, axis=1)
        predicted_class2 = jnp.argmax(preds2, axis=1)
        if save is not None:
            predicted_class = predicted_class1, predicted_class2
            self.save_fig(inputs, predicted_class, save)
        correct1 = predicted_class1 == target_class1
        correct2 = predicted_class2 == target_class2
        return jnp.mean(correct1 * correct2)

    def predict(self, params, inputs):
        logits_all = self.logit_predict(params, inputs)
        logits1, logits2 = jnp.split(logits_all, 2, -1)
        preds1 = logits1 - logsumexp(logits1, axis=1, keepdims=True)
        preds2 = logits2 - logsumexp(logits2, axis=1, keepdims=True)
        return preds1, preds2

    def run(self, data, index):
        (train_images, train_labels), (test_images, test_labels) = data
        batches = train_images, train_labels

        opt_init, opt_update, get_params = optimizers.adam(self.args.lr)

        @jit
        def update(i, opt_state, batch):
            params = get_params(opt_state)
            return opt_update(i, grad(self.loss)(params, batch), opt_state)

        rng = random.PRNGKey(npr.randint(1000000))
        _, init_params = self.init_random_params(rng, (-1, 2))
        opt_state = opt_init(init_params)
        itercount = itertools.count()

        params = get_params(opt_state)
        angles = [self.get_angle(params)]
        losses = [self.loss(params, batches)]

        step = self.args.steps
        if self.depth == 0:
            step *= 8
        elif self.depth == 1:
            step *= 4
        elif self.depth == 2:
            step *= 2

        for epoch in range(step):
            opt_state = update(next(itercount), opt_state, batches)
            params = get_params(opt_state)
            angles.append(self.get_angle(params))
            losses.append(self.loss(params, batches))

        params = get_params(opt_state)
        train_acc = self.accuracy(params, (train_images, train_labels))

        if self.args.plot_prediction:
            folder = os.path.join('lin_results', str(self.args.width),
                                  str(self.depth))
            os.makedirs(folder, exist_ok=True)
            fn = os.path.join(folder, str(index) + '.pdf')
            test_acc = self.accuracy(params, (test_images, test_labels), fn)
        else:
            test_acc = None
        return train_acc, test_acc, angles, losses


def plot_matrix(matrix, fn):
    for row in matrix:
        plt.plot(row)
    plt.savefig(fn)
    plt.clf()

def one_depth(args, depth, data):
    results = []
    angle_matrix = []
    loss_matrix = []
    for i in range(8):
        experiment = Experiment(args, depth)
        result = experiment.run(data, i)
        results.append(result[1])
        angle_matrix.append(result[2])
        loss_matrix.append(result[3])

    folder = os.path.join("lin_results", str(args.width), 'angles')
    os.makedirs(folder, exist_ok=True)
    fn = os.path.join(folder, str(depth) + 'd.pdf')
    plt.axhline(y=0, color='gray', linestyle='dashed')
    plt.axhline(y=90, color='gray', linestyle='dashed')
    plt.axhline(y=180, color='gray', linestyle='dashed')
    plot_matrix(angle_matrix, fn)

    folder = os.path.join("lin_results", str(args.width), 'loss')
    os.makedirs(folder, exist_ok=True)
    fn = os.path.join(folder, str(depth) + 'd.pdf')
    plot_matrix(loss_matrix, fn)

    return np.asarray(results)


def main(args):
    (train_images, train_labels), (test_images, test_labels) = get_data()

    train_labels = np.transpose(train_labels)
    test_labels = np.transpose(test_labels)

    n_values = 2
    train_labels1 = np.eye(n_values)[train_labels[0]]
    train_labels2 = np.eye(n_values)[train_labels[1]]
    train_labels = train_labels1, train_labels2
    test_labels1 = np.eye(n_values)[test_labels[0]]
    test_labels2 = np.eye(n_values)[test_labels[1]]
    test_labels = test_labels1, test_labels2
    data = (train_images, train_labels), (test_images, test_labels)

    all_results = []
    for depth in range(20):
        results = one_depth(args, depth, data)
        if args.plot_prediction:
            print(depth, results)
            all_results.append(results)
        else:
            print(depth)

    if args.plot_prediction:
        matrix = np.asarray(all_results)
        mean = np.mean(matrix, -1)
        std = np.std(matrix, -1)
        print(mean)
        print(std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='',
                        help='Log directory.')
    parser.add_argument('--steps', type=int, default=100,
                        help='Steps.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--width', type=int, default=32,
                        help='width.')
    parser.add_argument('--plot_prediction', action='store_true',
                        default=False,
                        help='Plot prediction.')
    main(parser.parse_args())
