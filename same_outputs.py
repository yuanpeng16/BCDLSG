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
from matplotlib import rc
import os
import argparse

import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu
from jax.scipy.special import logsumexp
import keras.datasets.mnist


class Experiment(object):
    def __init__(self, args):
        self.depth = args.depth

        layers = []
        for _ in range(self.depth - 1):
            layers += [Dense(args.width), Relu]
        layers.append(Dense(20))
        self.init_random_params, self.logit_predict = stax.serial(*layers)

    def loss(self, params, batch):
        inputs, targets = batch
        targets1, targets2 = targets, targets
        preds1, preds2 = self.predict(params, inputs)
        loss1 = -jnp.mean(jnp.sum(preds1 * targets1, axis=1))
        loss2 = -jnp.mean(jnp.sum(preds2 * targets2, axis=1))
        return loss1 + loss2

    def accuracy(self, params, batch):
        inputs, targets = batch
        targets1, targets2 = targets, targets
        target_class1 = jnp.argmax(targets1, axis=1)
        target_class2 = jnp.argmax(targets2, axis=1)
        preds1, preds2 = self.predict(params, inputs)
        predicted_class1 = jnp.argmax(preds1, axis=1)
        predicted_class2 = jnp.argmax(preds2, axis=1)
        correct1 = predicted_class1 == target_class1
        correct2 = predicted_class2 == target_class2
        correct = jnp.mean(correct1 * correct2)
        same = jnp.mean(predicted_class1 == predicted_class2)
        return correct, same

    def predict(self, params, inputs):
        logits_all = self.logit_predict(params, inputs)
        logits1, logits2 = jnp.split(logits_all, 2, -1)
        preds1 = logits1 - logsumexp(logits1, axis=1, keepdims=True)
        preds2 = logits2 - logsumexp(logits2, axis=1, keepdims=True)
        return preds1, preds2


def experiment(args):
    rng = random.PRNGKey(npr.randint(1000000))

    experiment = Experiment(args)

    step_size = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    (train_images, train_labels), (
        test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = np.reshape(train_images, [train_images.shape[0], -1])
    test_images = np.reshape(test_images, [test_images.shape[0], -1])

    train_images = train_images / 256 - 0.5
    test_images = test_images / 256 - 0.5
    random_images = np.random.rand(*test_images.shape) - 0.5

    n_values = np.max(train_labels) + 1
    train_labels = np.eye(n_values)[train_labels]
    test_labels = np.eye(n_values)[test_labels]

    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]

    batches = data_stream()

    opt_init, opt_update, get_params = optimizers.adam(step_size)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(experiment.loss)(params, batch), opt_state)

    _, init_params = experiment.init_random_params(rng, (-1, 28 * 28))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    train_acc_list = []
    train_same_list = []
    test_acc_list = []
    test_same_list = []
    random_same_list = []

    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))

        params = get_params(opt_state)
        train_acc, train_same = experiment.accuracy(params, (
            train_images, train_labels))
        test_acc, test_same = experiment.accuracy(params,
                                                  (test_images, test_labels))
        _, random_same = experiment.accuracy(params,
                                             (random_images, test_labels))
        train_acc_list.append(train_acc)
        train_same_list.append(train_same)
        test_acc_list.append(test_acc)
        test_same_list.append(test_same)
        random_same_list.append(random_same)
    lists = train_acc_list, train_same_list, test_acc_list, test_same_list, random_same_list
    return lists


def main(args):
    font = {'family': 'serif'}
    rc('font', **font)
    fontsize = 20

    train_acc_list = []
    train_same_list = []
    test_acc_list = []
    test_same_list = []
    random_same_list = []
    for _ in range(5):
        lists = experiment(args)
        train_acc_list.append(lists[0])
        train_same_list.append(lists[1])
        test_acc_list.append(lists[2])
        test_same_list.append(lists[3])
        random_same_list.append(lists[4])

    lists = train_acc_list, train_same_list, test_acc_list, test_same_list, random_same_list
    folder = os.path.join("same_outputs")
    os.makedirs(folder, exist_ok=True)
    fn = os.path.join(folder, 'same_outputs_' + str(args.depth) + '.pdf')
    names = ["Train acc", "Train same", "Test acc", "Test same", "Random same"]
    for x, name in zip(lists, names):
        x = 100 * np.asarray(x)
        mean = np.mean(x, 0)
        std = np.std(x, 0)
        plt.plot(mean, label=name)
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std,
                         alpha=0.2)
    plt.tick_params(labelsize=fontsize)
    if args.show_legend:
        plt.legend(prop={'size': fontsize})
    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel("Percentage (%)", fontsize=fontsize)
    plt.savefig(fn, bbox_inches='tight', pad_inches=0.01)
    plt.clf()

    fn = os.path.join(folder, 'random_same_' + str(args.depth) + '.txt')
    with open(fn, 'w') as f:
        for t, x in zip(test_acc_list, random_same_list):
            f.write(str(x[0]))
            f.write('\t')
            f.write(str(x[-1]))
            f.write('\t')
            f.write(str(t[-1]))
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--depth', type=int, default=8,
                        help='Depth.')
    parser.add_argument('--width', type=int, default=128,
                        help='Width.')
    parser.add_argument('--show_legend', action='store_true',
                        default=False, help='Show legend.')
    main(parser.parse_args())
