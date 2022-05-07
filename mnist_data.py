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

        samples1 = self._get_data(args.dataset1)
        if args.dataset1 == args.dataset2:
            samples2 = samples1
        else:
            samples2 = self._get_data(args.dataset2)
        self.train_samples1, self.test_samples1, self.shape1 = samples1
        self.train_samples2, self.test_samples2, self.shape2 = samples2

        self.train_label_pairs = []
        self.test_label_pairs = []
        self.get_label_splits()

    def _get_data(self, data_name):
        if data_name == 'mnist':
            dataset = tf.keras.datasets.mnist
        elif data_name == 'cifar10':
            dataset = tf.keras.datasets.cifar10
        elif data_name == 'fashion_mnist':
            dataset = tf.keras.datasets.fashion_mnist
        else:
            assert False

        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        if len(x_train.shape) == 3:
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)
        shape = x_train.shape[1:]
        train_samples = self._prepare_data(x_train, y_train)
        test_samples = self._prepare_data(x_test, y_test)
        return train_samples, test_samples, shape

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

    def _get_samples(self, samples1, samples2, k, is_train):
        x_list, y_list, y2_list = [], [], []
        if is_train:
            label_list = random.choices(self.train_label_pairs, k=k)
        else:
            label_list = random.choices(self.test_label_pairs, k=k)

        for y, y2 in label_list:
            x = self._merge(y, y2, samples1, samples2)
            x_list.append(x)
            y_list.append(one_hot(y, self.output_nodes))
            y2_list.append(one_hot(y2, self.output_nodes))
        x_list = np.asarray(x_list)
        y_list = np.asarray(y_list)
        y2_list = np.asarray(y2_list)
        return x_list, [y_list, y2_list]

    def get_training_samples(self, k):
        return self._get_samples(self.train_samples1, self.train_samples2, k,
                                 is_train=True)

    def get_eval_samples(self, k):
        return self.get_training_samples(k)

    def get_test_samples(self, k, randomize=False):
        samples, y_list = self._get_samples(
            self.test_samples1, self.test_samples2, k, is_train=False)
        if randomize:
            samples = np.random.rand(*samples.shape)
        return samples, y_list

    def _merge(self, y, y2, samples1, samples2):
        pass


class LongDataGenerator(RandomDataGenerator):
    def _merge(self, y, y2, samples1, samples2):
        x1 = random.choice(samples1[y])
        x = [0 * x1] * self.output_nodes
        x[y2] = x1
        x = np.concatenate(x, axis=1)
        return x

    def get_input_shape(self):
        return tuple(np.multiply(self.shape1, [1, self.output_nodes, 1]))


class PairedDataGenerator(RandomDataGenerator):
    def _merge(self, y, y2, samples1, samples2):
        x1 = random.choice(samples1[y])
        x2 = random.choice(samples2[y2])
        assert self.shape1[0] == self.shape2[0]
        assert self.shape1[2] == self.shape2[2]
        x = np.concatenate((x1, x2), axis=1)
        return x

    def get_input_shape(self):
        return self.shape1[0],\
               self.shape1[1] + self.shape2[1], self.shape1[2]


class StackedDataGenerator(RandomDataGenerator):
    def _merge(self, y, y2, samples1, samples2):
        x1 = random.choice(samples1[y])
        x2 = random.choice(samples2[y2])
        assert self.shape1[0] == self.shape2[0]
        assert self.shape1[1] == self.shape2[1]
        x = np.concatenate((x1, x2), axis=-1)
        return x

    def get_input_shape(self):
        return self.shape1[0],\
               self.shape1[1], self.shape1[2] + self.shape2[2]


class AddedDataGenerator(RandomDataGenerator):
    def _merge(self, y, y2, samples1, samples2):
        x1 = random.choice(samples1[y])
        x2 = random.choice(samples2[y2])
        assert self.shape1[0] == self.shape2[0]
        assert self.shape1[1] == self.shape2[1]
        assert self.shape1[2] == self.shape2[2]
        x = 0.5 * (x1 + x2)
        return x

    def get_input_shape(self):
        return self.shape1
