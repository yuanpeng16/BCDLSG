import tensorflow as tf
import numpy as np
import random


def get_data_generator(args):
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
    return dg


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
        x = 0.5 * (x1 + x2)
        return x

    def get_input_shape(self):
        return self.shape
