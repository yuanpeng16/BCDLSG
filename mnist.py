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


def one_hot(a, output_nodes):
    ret = [0] * output_nodes
    ret[a] = 1
    return ret


class RandomDataGenerator(object):
    def __init__(self, args):
        self.args = args
        self.output_nodes = 10

        if args.dataset == 'mnist':
            dataset = tf.keras.datasets.mnist
        elif args.dataset == 'cifar10':
            dataset = tf.keras.datasets.cifar10
        elif args.dataset == 'fashion_mnist':
            dataset = tf.keras.datasets.fashion_mnist
        else:
            assert False

        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        if len(x_train.shape) == 3:
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)
        self.shape = x_train.shape[1:]

        self.train_samples = self._prepare_data(x_train, y_train)
        self.test_samples = self._prepare_data(x_test, y_test)
        self.train_label_pairs = []
        self.test_label_pairs = []
        self.get_label_splits()

    def _prepare_data(self, x_all, y_all):
        assert len(x_all) == len(y_all)
        x_all = x_all / 255.0
        x_all = x_all.astype("float32")

        data = [[] for _ in range(self.output_nodes)]
        for x, y in zip(x_all, y_all):
            y = int(y)
            data[y].append(x)
        return data

    def is_train_label(self, x, y):
        return x < 5 or y < 5

    def is_train_label2(self, x, y):
        diff = (y - x + self.output_nodes) % self.output_nodes
        return diff < 2

    def get_label_splits(self):
        for i in range(self.output_nodes):
            for j in range(self.output_nodes):
                if self.is_train_label(i, j):
                    self.train_label_pairs.append((i, j))
                else:
                    self.test_label_pairs.append((i, j))

    def get_test_label_pairs(self):
        return self.test_label_pairs

    def get_output_nodes(self):
        return self.output_nodes

    def _get_samples(self, samples, k, is_train):
        x_list, y_list, y2_list = [], [], []
        if is_train:
            label_list = random.choices(self.train_label_pairs, k=k)
        else:
            label_list = random.choices(self.test_label_pairs, k=k)

        for y, y2 in label_list:
            x = self._merge(y, y2, samples)
            x_list.append(x)
            y_list.append(one_hot(y, self.output_nodes))
            y2_list.append(one_hot(y2, self.output_nodes))
        x_list = np.asarray(x_list)
        y_list = np.asarray(y_list)
        y2_list = np.asarray(y2_list)
        return x_list, [y_list, y2_list]

    def get_training_samples(self, k):
        return self._get_samples(self.train_samples, k, is_train=True)

    def get_eval_samples(self, k):
        return self.get_training_samples(k)

    def get_test_samples(self, k, randomize=False):
        samples, y_list = self._get_samples(self.test_samples, k,
                                            is_train=False)
        if randomize:
            samples = np.random.rand(*samples.shape)
        return samples, y_list

    def _merge(self, y, y2, samples):
        pass


class LongDataGenerator(RandomDataGenerator):
    def _merge(self, y, y2, samples):
        x1 = random.choice(samples[y])
        x = [0 * x1] * self.output_nodes
        x[y2] = x1
        x = np.concatenate(x, axis=1)
        return x

    def get_input_shape(self):
        return tuple(np.multiply(self.shape, [1, self.output_nodes, 1]))


class PairedDataGenerator(RandomDataGenerator):
    def _merge(self, y, y2, samples):
        x1 = random.choice(samples[y])
        x2 = random.choice(samples[y2])
        x = np.concatenate((x1, x2), axis=1)
        return x

    def get_input_shape(self):
        return tuple(np.multiply(self.shape, [1, 2, 1]))


class StackedDataGenerator(RandomDataGenerator):
    def _merge(self, y, y2, samples):
        x1 = random.choice(samples[y])
        x2 = random.choice(samples[y2])
        x = np.concatenate((x1, x2), axis=-1)
        return x

    def get_input_shape(self):
        return tuple(np.multiply(self.shape, [1, 1, 2]))


class AddedDataGenerator(RandomDataGenerator):
    def _merge(self, y, y2, samples):
        x1 = random.choice(samples[y])
        x2 = random.choice(samples[y2])
        x = x1 + x2
        return x

    def get_input_shape(self):
        return self.shape

class DeepModelGenerator(object):
    def __init__(self, args, input_shape, output_nodes):
        self.args = args
        self.input_shape = input_shape
        self.output_nodes = output_nodes

    def get_structure(self):
        inputs = Input(shape=self.input_shape)
        x = inputs
        for _ in range(1):
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        for _ in range(2):
            x = tf.keras.layers.Dense(64, activation='relu')(x)

        if self.args.loss_type == 'hinge':
            activation = 'linear'
        else:
            activation = 'softmax'
        outputs = [
            Dense(self.output_nodes, activation=activation, name='y1')(x),
            Dense(self.output_nodes, activation=activation, name='y2')(x)
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
    def __init__(self, args, model, datasets, test_label_pairs):
        self.args = args
        self.model = model
        self.datasets = datasets
        self.test_label_pairs = set(test_label_pairs)

    def evaluate(self, x, y):
        y_hat = self.model(x)
        n_samples = len(y[0])
        hit1, hit2, hit, sg_hit = 0, 0, 0, 0
        for i in range(n_samples):
            y1_hat = np.argmax(y_hat[0][i])
            y2_hat = np.argmax(y_hat[1][i])
            if (y1_hat, y2_hat) in self.test_label_pairs:
                sg_hit += 1
            h1 = y[0][i][y1_hat] == 1
            h2 = y[1][i][y2_hat] == 1
            if h1:
                hit1 += 1
            if h2:
                hit2 += 1
            if h1 and h2:
                hit += 1
        acc = hit / n_samples
        acc1 = hit1 / n_samples
        acc2 = hit2 / n_samples
        sg_acc = sg_hit / n_samples
        return acc1, acc2, acc, sg_acc

    def test_evaluate(self, x, y):
        return self.evaluate(x, y)

    def evaluate_all(self):
        ret = []
        ret.extend(
            self.evaluate(self.datasets[0][0], self.datasets[0][1]))
        ret.append("\t")
        ret.extend(
            self.test_evaluate(self.datasets[1][0], self.datasets[1][1]))
        return ret


class AdversarialEvaluator(Evaluator):
    def __init__(self, args, model, datasets, test_label_pairs):
        super().__init__(args, model, datasets, test_label_pairs)
        self.initialize()

    def initialize(self):
        self.loss_object_list = [tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction='none') for _ in range(2)]

    def one_step(self, x, y, optimizer):
        with tf.GradientTape() as tape:
            y_hat = self.model(x)
            loss_list = [loss_object(labels, predictions) for
                         loss_object, labels, predictions in
                         zip(self.loss_object_list, y, y_hat)]
            losses = sum(loss_list)
            reduced_loss = tf.reduce_sum(losses)
        gradients = tape.gradient(reduced_loss, x)
        optimizer.apply_gradients(zip([gradients], [x]))
        return x

    def test_evaluate(self, x, y):
        result = self.evaluate(x, y)
        x = tf.Variable(x, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(lr=self.args.lr * 100)
        for _ in range(10):
            x = self.one_step(x, y, optimizer)
        result2 = self.evaluate(x, y)
        return list(result), '\t', list(result2)


class RandomAdversarialEvaluator(AdversarialEvaluator):
    def initialize(self):
        self.output_nodes = [
            self.model.outputs[0].shape[-1],
            self.model.outputs[1].shape[-1]
        ]
        self.loss_object = []
        for output_node in self.output_nodes:
            self.loss_object.append([tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, reduction='none') for _ in
                range(output_node)])

    def one_step(self, x, y, optimizer):
        loss_list = []
        loss_candidates = []
        with tf.GradientTape() as tape:
            y_hat = self.model(x)
            for los, labels, predictions in zip(self.loss_object,
                                                y, y_hat):
                output_node = len(los)
                losses = []
                for i in range(output_node):
                    labels = labels * 0 + one_hot(i, output_node)
                    loss = los[i](labels, predictions)
                    losses.append(loss)
                loss_list.append(losses)
            for y0, y1 in self.test_label_pairs:
                added_loss = loss_list[0][y0] + loss_list[1][y1]
                loss_candidates.append(tf.expand_dims(added_loss, -1))
            loss_candidates = tf.concat(loss_candidates, 1)
            min_loss = tf.reduce_min(loss_candidates, axis=-1)
            best_loss = tf.reduce_sum(min_loss)
        gradients = tape.gradient(best_loss, x)
        optimizer.apply_gradients(zip([gradients], [x]))
        return x


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
    elif args.merge_type == 'added':
        dg = AddedDataGenerator(args)
    else:
        assert False

    randomize = args.test_distribution == 'random'
    eval_data = dg.get_eval_samples(100)
    test_data = dg.get_test_samples(100, randomize=randomize)

    if args.save_image:
        for i in range(5):
            save_image(eval_data[0][i], 'eval_' + str(i) + '.png')
            save_image(test_data[0][i], 'test_' + str(i) + '.png')
        return

    output_nodes = dg.get_output_nodes()
    mg = DeepModelGenerator(args, dg.get_input_shape(), output_nodes)
    model = mg.get_model()
    test_label_pairs = dg.get_test_label_pairs()
    if args.adversarial:
        if args.any_generalization:
            ev = RandomAdversarialEvaluator(
                args, model, [eval_data, test_data], test_label_pairs)
        else:
            ev = AdversarialEvaluator(
                args, model, [eval_data, test_data], test_label_pairs)
    else:
        ev = Evaluator(args, model, [eval_data, test_data], test_label_pairs)

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
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--steps', type=int, default=500,
                        help='Steps.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--merge_type', type=str, default='added',
                        help='Merge type.')
    parser.add_argument('--test_distribution', type=str, default='original',
                        help='Test distribution.')
    parser.add_argument('--random_threshold', type=float, default=0.85,
                        help='Threshold to randomize the second input.')
    parser.add_argument('--save_image', action='store_true', default=False,
                        help='Show image and stop.')
    parser.add_argument('--compute_gradient', action='store_true',
                        default=False, help='Compute gradient.')
    parser.add_argument('--adversarial', action='store_true',
                        default=False,
                        help='Use adversarial learning on test.')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Test Dataset.')
    parser.add_argument('--any_generalization', action='store_true',
                        default=False, help='Any systematic generalization.')
    main(parser.parse_args())
