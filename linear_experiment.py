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

import time
import itertools

import numpy as np
import numpy.random as npr

import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
from jax.scipy.special import logsumexp
import keras.datasets.mnist


# from examples import datasets


def get_data():
    train_inputs = np.asarray([
        [-1, 1],
        [-1, -1],
        [1, -1]
    ], dtype=np.float32)
    train_labels = np.asarray([
        [0, 1],
        [0, 0],
        [1, 0]
    ], dtype=np.int32)

    test_inputs = np.asarray([
        [1, 1]
    ], dtype=np.float32)
    test_labels = np.asarray([
        [1, 1]
    ], dtype=np.int32)
    return (train_inputs, train_labels), (test_inputs, test_labels)


rng = random.PRNGKey(0)


class Experiment(object):
    def __init__(self, width, depth):
        layers = []
        for _ in range(depth):
            layers += [Dense(width), Relu]
        layers.append(Dense(4))
        self.init_random_params, self.logit_predict = stax.serial(*layers)

    def loss(self, params, batch):
        inputs, targets = batch
        targets1, targets2 = targets
        preds1, preds2 = self.predict(params, inputs)
        loss1 = -jnp.mean(jnp.sum(preds1 * targets1, axis=1))
        loss2 = -jnp.mean(jnp.sum(preds2 * targets2, axis=1))
        return loss1 + loss2

    def accuracy(self, params, batch):
        inputs, targets = batch
        targets1, targets2 = targets
        target_class1 = jnp.argmax(targets1, axis=1)
        target_class2 = jnp.argmax(targets2, axis=1)
        preds1, preds2 = self.predict(params, inputs)
        predicted_class1 = jnp.argmax(preds1, axis=1)
        predicted_class2 = jnp.argmax(preds2, axis=1)
        correct1 = predicted_class1 == target_class1
        correct2 = predicted_class2 == target_class2
        return jnp.mean(correct1 * correct2)

    def predict(self, params, inputs):
        logits_all = self.logit_predict(params, inputs)
        logits1, logits2 = jnp.split(logits_all, 2, -1)
        preds1 = logits1 - logsumexp(logits1, axis=1, keepdims=True)
        preds2 = logits2 - logsumexp(logits2, axis=1, keepdims=True)
        return preds1, preds2

    def run(self, data):
        step_size = 0.001
        num_epochs = 100

        (train_images, train_labels), (test_images, test_labels) = data
        batches = train_images, train_labels

        opt_init, opt_update, get_params = optimizers.adam(step_size)

        @jit
        def update(i, opt_state, batch):
            params = get_params(opt_state)
            return opt_update(i, grad(self.loss)(params, batch), opt_state)

        _, init_params = self.init_random_params(rng, (-1, 2))
        opt_state = opt_init(init_params)
        itercount = itertools.count()

        for epoch in range(num_epochs):
            opt_state = update(next(itercount), opt_state, batches)

        params = get_params(opt_state)
        train_acc = self.accuracy(params, (train_images, train_labels))
        test_acc = self.accuracy(params, (test_images, test_labels))
        return train_acc, test_acc


def one_depth(depth, data):
    results = []
    for i in range(20):
        experiment = Experiment(1024, depth)
        result = experiment.run(data)
        results.append(result[1])
    return np.asarray(results)


def main():
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
    for depth in range(1, 8):
        results = one_depth(depth, data)
        print(depth, results)
        all_results.append(results)

    matrix = np.asarray(all_results)
    mean = np.mean(matrix, -1)
    std = np.std(matrix, -1)
    print(mean)
    print(std)


if __name__ == "__main__":
    main()
