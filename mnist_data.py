import tensorflow as tf
import numpy as np
import random
from PIL import Image


def get_data_generator(args):
    if args.merge_type == 'slide':
        dg = LongDataGenerator(args)
    elif args.merge_type == 'paired':
        dg = PairedDataGenerator(args)
    elif args.merge_type == 'stacked':
        dg = StackedDataGenerator(args)
    elif args.merge_type == 'added':
        dg = AddedDataGenerator(args)
    elif args.merge_type == 'max':
        dg = MaxDataGenerator(args)
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

        train1, test1, self.shape1 = self._get_data(args.dataset1)
        train2, test2, self.shape2 = self._get_data(args.dataset2)
        self.input_shape = self.compute_input_shape()

        if self.args.input_permutation:
            size = np.prod(self.input_shape)
            self.permutation_mapping = list(range(size))
            random.shuffle(self.permutation_mapping)

        # Preprocessing
        self.train_samples1 = self._prepare_data(train1, False)
        self.test_samples1 = self._prepare_data(test1, False)
        self.train_samples2 = self._prepare_data(train2, True)
        self.test_samples2 = self._prepare_data(test2, True)

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
        train_samples = [x_train, y_train]
        test_samples = [x_test, y_test]
        return train_samples, test_samples, shape

    def resize(self, x, size, rotate):
        if not rotate and x.shape[0] == size[0] and x.shape[1] == size[1]:
            return x
        if x.shape[-1] == 1:
            x = np.squeeze(x)
        x = Image.fromarray(np.uint8(255 * x))
        if rotate:
            x = x.rotate(90)
        if x.size[0] != size[0] or x.size[1] != size[1]:
            x = x.resize(size)
        x = np.array(x) / 255.0
        if len(x.shape) < 3:
            x = np.expand_dims(x, -1)
        return x

    def _prepare_data(self, data, rotate):
        x_all, y_all = data
        assert len(x_all) == len(y_all)
        x_all = x_all / 255.0
        x_all = x_all.astype("float32")

        size = self.compute_one_input_shape()[:2]
        data = [[] for _ in range(self.output_nodes)]
        for x, y in zip(x_all, y_all):
            y = int(y)
            x = self.resize(x, size, rotate)
            data[y].append(x)
        return data

    def get_input_shape(self):
        return self.input_shape

    def is_train_label(self, x, y):
        if self.args.label_split == 'tile':
            return x < 5 or y < 5
        elif self.args.label_split == 'one_shot':
            return x == 0 or y == 0
        elif self.args.label_split == 'many_shot':
            return x < 9 or y < 9
        elif self.args.label_split == 'diagonal':
            diff = (y - x + self.output_nodes) % self.output_nodes
            return diff < 5
        elif self.args.label_split == 'one_shot_diagonal':
            return x == y
        elif self.args.label_split == 'many_shot_diagonal':
            return x != y
        elif self.args.label_split == 'one_label':
            return x < 9 or y < 1
        assert False

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

    def _permutate(self, x):
        shape = x.shape
        x_flat = np.reshape(x, [-1])
        y_flat = [
            x_flat[self.permutation_mapping[i]] for i in range(len(x_flat))]
        y = np.reshape(y_flat, shape)
        return y

    def _get_samples(self, samples1, samples2, k, is_train):
        x_list, y_list, y2_list = [], [], []
        if is_train:
            label_list = random.choices(self.train_label_pairs, k=k)
        else:
            label_list = random.choices(self.test_label_pairs, k=k)

        for y, y2 in label_list:
            x = self._merge(y, y2, samples1, samples2)
            if self.args.input_permutation:
                x = self._permutate(x)
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

    def compute_one_input_shape(self):
        s1 = max(self.shape1[0], self.shape2[1])
        s2 = max(self.shape1[1], self.shape2[0])
        s3 = max(self.shape1[2], self.shape2[2])
        return s1, s2, s3

    def _merge(self, y, y2, samples1, samples2):
        pass

    def compute_input_shape(self):
        pass


class LongDataGenerator(RandomDataGenerator):
    def _merge(self, y, y2, samples1, samples2):
        x1 = random.choice(samples1[y])
        x = [0 * x1] * self.output_nodes
        x[y2] = x1
        x = np.concatenate(x, axis=1)
        return x

    def compute_input_shape(self):
        return tuple(np.multiply(self.shape1, [1, self.output_nodes, 1]))


class PairedDataGenerator(RandomDataGenerator):
    def _merge(self, y, y2, samples1, samples2):
        x1 = random.choice(samples1[y])
        x2 = random.choice(samples2[y2])
        assert x1.shape[0] == x2.shape[0]
        if x1.shape[2] != x2.shape[2]:
            if x1.shape[2] == 1:
                x1 = np.broadcast_to(x1, x2.shape)
            elif x2.shape[2] == 1:
                x2 = np.broadcast_to(x2, x1.shape)
            else:
                assert False
        x = np.concatenate((x1, x2), axis=1)
        return x

    def compute_input_shape(self):
        shape = self.compute_one_input_shape()
        return tuple(np.multiply(shape, [1, 2, 1]))


class StackedDataGenerator(RandomDataGenerator):
    def _merge(self, y, y2, samples1, samples2):
        x1 = random.choice(samples1[y])
        x2 = random.choice(samples2[y2])
        assert x1.shape[0] == x2.shape[0]
        assert x1.shape[1] == x2.shape[1]
        x = np.concatenate((x1, x2), axis=-1)
        return x

    def compute_input_shape(self):
        shape = self.compute_one_input_shape()
        depth = self.shape1[2] + self.shape2[2]
        return shape[0], shape[1], depth


class AddedDataGenerator(RandomDataGenerator):
    def overlap(self, x1, x2):
        return 0.5 * (x1 + x2)

    def _merge(self, y, y2, samples1, samples2):
        x1 = random.choice(samples1[y])
        x2 = random.choice(samples2[y2])
        assert x1.shape[0] == x2.shape[0]
        assert x1.shape[1] == x2.shape[1]
        assert x1.shape[2] == x2.shape[2] \
               or x1.shape[2] == 1 or x2.shape[2] == 1
        return self.overlap(x1, x2)

    def compute_input_shape(self):
        return self.compute_one_input_shape()


class MaxDataGenerator(AddedDataGenerator):
    def overlap(self, x1, x2):
        return np.maximum(x1, x2)
