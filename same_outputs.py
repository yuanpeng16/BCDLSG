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
import matplotlib.pyplot as plt
import os
import argparse

import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
from jax.scipy.special import logsumexp
import keras.datasets.mnist


class Experiment(object):
    def __init__(self):
        self.depth = 4

        layers = []
        for _ in range(self.depth):
            layers += [Dense(128)]
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


if __name__ == "__main__":

  experiment = Experiment()

  rng = random.PRNGKey(0)

  step_size = 0.001
  num_epochs = 100
  batch_size = 256

  (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
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

  print("\nStarting training...")
  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      opt_state = update(next(itercount), opt_state, next(batches))
    epoch_time = time.time() - start_time

    params = get_params(opt_state)
    train_acc, train_same = experiment.accuracy(params, (train_images, train_labels))
    test_acc, test_same = experiment.accuracy(params, (test_images, test_labels))
    _, random_same = experiment.accuracy(params,(random_images, test_labels))
    print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
    print(f"Training set accuracy {train_acc}")
    print(f"Test set accuracy {test_acc}")
    print(f"Training set same {train_same}")
    print(f"Test set same {test_same}")
    print(f"Random set same {random_same}")
