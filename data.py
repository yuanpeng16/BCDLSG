import tensorflow as tf
import numpy as np
import random
from PIL import Image

from zeroshot_data import APYDataGenerator
from zeroshot_data import AWA2DataGenerator
from zeroshot_data import CUBDataGenerator
from zeroshot_data import SUNDataGenerator


def get_data_generator(args):
    if args.merge_type == 'slide':
        dg = LongDataGenerator(args)
    elif args.merge_type == 'paired':
        dg = PairedDataGenerator(args)
    elif args.merge_type == 'stacked':
        dg = StackedDataGenerator(args)
    elif args.merge_type == 'added':
        dg = AddedDataGenerator(args)
    elif args.merge_type == 'average':
        dg = AverageDataGenerator(args)
    elif args.merge_type == 'max':
        dg = MaxDataGenerator(args)
    elif args.merge_type == 'text':
        dg = TextDataGenerator(args)
    elif args.merge_type == 'length':
        dg = LengthDataGenerator(args)
    elif args.merge_type == 'zero_shot_apy':
        dg = APYDataGenerator(args)
    elif args.merge_type == 'zeroshot_awa2':
        dg = AWA2DataGenerator(args)
    elif args.merge_type == 'zeroshot_cub':
        dg = CUBDataGenerator(args)
    elif args.merge_type == 'zeroshot_sun':
        dg = SUNDataGenerator(args)
    elif args.merge_type == 'single':
        dg = SingleDataGenerator(args)
    else:
        raise ValueError(
            '{0} is not a valid merge_type.'.format(args.merge_type))
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
        rotate = args.rotate_second_input
        if rotate:
            self.shape2 = (self.shape2[1], self.shape2[0], self.shape2[2])
        self.input_shape = self.compute_input_shape()

        if self.args.input_permutation:
            size = np.prod(self.input_shape)
            self.permutation_mapping = list(range(size))
            random.shuffle(self.permutation_mapping)

        # Preprocessing
        self.train_samples1 = self._prepare_data(train1, False)
        self.test_samples1 = self._prepare_data(test1, False)
        self.train_samples2 = self._prepare_data(train2, rotate)
        self.test_samples2 = self._prepare_data(test2, rotate)

        self.train_label_pairs = []
        self.test_label_pairs = []
        self.get_label_splits()
        self.all_train_pairs = self.train_label_pairs + self.test_label_pairs

    def _get_data(self, data_name):
        if data_name == 'mnist':
            dataset = tf.keras.datasets.mnist
        elif data_name == 'cifar10':
            dataset = tf.keras.datasets.cifar10
        elif data_name == 'fashion_mnist':
            dataset = tf.keras.datasets.fashion_mnist
        else:
            raise ValueError('{0} is not a valid data_name.'.format(data_name))

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
        x = Image.fromarray(np.uint8(255 * (x + 0.5)))
        if rotate:
            x = x.rotate(90)
        if x.size[0] != size[0] or x.size[1] != size[1]:
            x = x.resize(size)
        x = np.array(x) / 255.0 - 0.5
        if len(x.shape) < 3:
            x = np.expand_dims(x, -1)
        return x

    def _prepare_data(self, data, rotate):
        x_all, y_all = data
        assert len(x_all) == len(y_all)
        x_all = x_all / 255.0 - 0.5
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
        elif self.args.label_split == 'many_label':
            return x < 5 or y < 1
        raise ValueError(
            '{0} is not a valid label_split.'.format(self.args.label_split))

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

    def _permute(self, x):
        shape = x.shape
        x_flat = np.reshape(x, [-1])
        y_flat = [
            x_flat[self.permutation_mapping[i]] for i in range(len(x_flat))]
        y = np.reshape(y_flat, shape)
        return y

    def _get_samples(self, samples1, samples2, k, is_train, pretrain=False):
        x_list, y_list, y2_list = [], [], []
        if is_train:
            if pretrain:
                label_list = random.choices(self.all_train_pairs, k=k)
            else:
                label_list = random.choices(self.train_label_pairs, k=k)
        else:
            label_list = random.choices(self.test_label_pairs, k=k)

        for y, y2 in label_list:
            x = self._merge(y, y2, samples1, samples2)
            if self.args.input_permutation:
                x = self._permute(x)
            x_list.append(x)
            y_list.append(one_hot(y, self.output_nodes))
            y2_list.append(one_hot(y2, self.output_nodes))
        x_list = np.asarray(x_list)
        y_list = np.asarray(y_list)
        y2_list = np.asarray(y2_list)
        return x_list, [y_list, y2_list]

    def get_training_samples(self, k, pretrain=False):
        return self._get_samples(self.train_samples1, self.train_samples2, k,
                                 is_train=True, pretrain=pretrain)

    def get_eval_samples(self, k):
        return self.get_training_samples(k)

    def get_test_samples(self, k, randomize=False):
        samples, y_list = self._get_samples(
            self.test_samples1, self.test_samples2, k, is_train=False)
        if randomize:
            samples = np.random.rand(*samples.shape) - 0.5
        return samples, y_list

    def compute_one_input_shape(self):
        return tuple(np.maximum(self.shape1, self.shape2))

    def get_vocab_size(self):
        return 0

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


class TextDataGenerator(RandomDataGenerator):
    def __init__(self, args):
        self.args = args
        if args.dataset1 == 'imdb':
            self.output_nodes = 2
        else:
            self.output_nodes = 10

        self.maxlen = 200
        self.max_length = 2 * self.maxlen + 1
        train1, test1, self.shape1 = self._get_data(args.dataset1, self.maxlen)
        train2, test2, self.shape2 = self._get_data(args.dataset2, self.maxlen)
        self.input_shape = self.compute_input_shape()

        inputs = [train1[0], test1[0], train2[0], test2[0]]
        self.vocab_size = max(max([max(x) for x in xs]) for xs in inputs) + 1

        # Preprocessing
        self.train_samples1 = self._prepare_data(train1)
        self.test_samples1 = self._prepare_data(test1)
        self.train_samples2 = self._prepare_data(train2)
        self.test_samples2 = self._prepare_data(test2)

        self.train_label_pairs = []
        self.test_label_pairs = []
        self.get_label_splits()

    def _filter(self, x_list, y_list, label_map):
        x_ret = []
        y_ret = []
        for x, y in zip(x_list, y_list):
            if y in label_map:
                x_ret.append(x)
                y_ret.append(label_map[y])
        return np.asarray(x_ret), np.asarray(y_ret)

    def _convert(self, data):
        (x_train, y_train), (x_test, y_test) = data
        stat = {}
        for y in y_train:
            stat[y] = stat.get(y, 0) + 1
        order = sorted(list(stat.items()), key=lambda x: x[1], reverse=True)
        label_map = {x[0]: i for i, x in enumerate(order[:self.output_nodes])}

        x_train, y_train = self._filter(x_train, y_train, label_map)
        x_test, y_test = self._filter(x_test, y_test, label_map)
        return (x_train, y_train), (x_test, y_test)

    def _get_data(self, data_name, maxlen):
        if data_name == 'imdb':
            dataset = tf.keras.datasets.imdb
        elif data_name == 'reuters':
            dataset = tf.keras.datasets.reuters
        else:
            raise ValueError('{0} is not a valid data_name.'.format(data_name))

        (x_train, y_train), (x_test, y_test) = dataset.load_data(maxlen=maxlen)
        if data_name == 'reuters':
            data = (x_train, y_train), (x_test, y_test)
            (x_train, y_train), (x_test, y_test) = self._convert(data)

        shape = x_train.shape[1:]
        train_samples = [x_train, y_train]
        test_samples = [x_test, y_test]
        return train_samples, test_samples, shape

    def compute_input_shape(self):
        return (self.max_length,)

    def _prepare_data(self, data):
        x_all, y_all = data
        assert len(x_all) == len(y_all)
        data = [[] for _ in range(self.output_nodes)]
        for x, y in zip(x_all, y_all):
            y = int(y)
            data[y].append(x)
        return data

    def is_train_label(self, x, y):
        if self.output_nodes == 10:
            return super().is_train_label(x, y)
        if self.args.label_split == 'tile':
            return x < 1 or y < 1
        elif self.args.label_split == 'diagonal':
            return x != y
        raise ValueError(
            '{0} is not a valid label_split.'.format(self.args.merge_type))

    def get_vocab_size(self):
        # Dedicated method for text data.
        return self.vocab_size

    def _merge(self, y, y2, samples1, samples2):
        x1 = random.choice(samples1[y])
        x2 = random.choice(samples2[y2])
        x = x1 + x2
        padded_x = np.array(x + ([0] * (self.max_length - len(x))))
        return padded_x

    def _get_random_input(self):
        length1 = np.random.randint(1, high=self.maxlen)
        length2 = np.random.randint(1, high=self.maxlen)
        x1 = [1] + np.random.randint(2, high=self.vocab_size,
                                     size=length1).tolist()
        x2 = [1] + np.random.randint(2, high=self.vocab_size,
                                     size=length2).tolist()
        x = x1 + x2
        padding = [0] * (self.max_length - len(x))
        x = x + padding
        return x

    def get_test_samples(self, k, randomize=False):
        samples, y_list = self._get_samples(
            self.test_samples1, self.test_samples2, k, is_train=False)
        if randomize:
            random_samples = []
            for i in range(samples.shape[0]):
                random_samples.append(self._get_random_input())
            samples = np.array(random_samples)
        return samples, y_list


class SingleDataGenerator(RandomDataGenerator):
    def __init__(self, args):
        self.args = args
        self.output_nodes = 10

        train1, test1, self.shape1 = self._get_data(args.dataset1)
        self.input_shape = self.compute_input_shape()

        # Preprocessing
        self.train_samples1 = self._prepare_data(train1, False)
        self.test_samples1 = self._prepare_data(test1, False)
        self.train_samples2 = self.train_samples1
        self.test_samples2 = self.test_samples1

        self.train_label_pairs = [(i, i) for i in range(5)]
        self.test_label_pairs = [(i, i) for i in range(5, 10)]
        self.all_train_pairs = self.train_label_pairs + self.test_label_pairs

    def compute_one_input_shape(self):
        return self.shape1

    def compute_input_shape(self):
        return self.shape1

    def _get_samples(self, samples1, samples2, k, is_train, pretrain=False):
        x_list, [y_list, _] = super()._get_samples(
            samples1, samples2, k, is_train, pretrain=pretrain)
        return x_list, y_list

    def _merge(self, y, y2, samples1, samples2):
        x1 = random.choice(samples1[y])
        return x1


class AverageDataGenerator(RandomDataGenerator):
    def __init__(self, args):
        self.args = args
        self.output_nodes = 10

        train1, test1, self.shape1 = self._get_data(args.dataset1)
        train2, test2, self.shape2 = train1, test1, self.shape1
        rotate = args.rotate_second_input
        if rotate:
            self.shape2 = (self.shape2[1], self.shape2[0], self.shape2[2])
        self.input_shape = self.compute_input_shape()

        if self.args.input_permutation:
            size = np.prod(self.input_shape)
            self.permutation_mapping = list(range(size))
            random.shuffle(self.permutation_mapping)

        train = [np.concatenate([train1[i], test1[i]], 0) for i in range(2)]

        # Preprocessing
        self.train_samples1 = self._prepare_data(train, False)
        self.test_samples1 = self.train_samples1
        self.train_samples2 = self.train_samples1
        self.test_samples2 = self.test_samples1

        self.train_label_pairs = []
        self.test_label_pairs = []
        self.get_label_splits()
        self.all_train_pairs = self.train_label_pairs + self.test_label_pairs

    def _prepare_data(self, data, rotate):
        x_all, y_all = data
        assert len(x_all) == len(y_all)
        x_all = x_all / 255.0 - 0.5
        x_all = x_all.astype("float32")

        a_all = [np.average(x) for x in x_all]
        min_a = min(a_all)
        max_a = max(a_all)
        delta = 0.1 * (max_a - min_a)
        max_a -= delta
        min_a += delta

        data = [[[] for _ in range(self.output_nodes)] for _ in
                range(self.output_nodes)]
        for x, y, a in zip(x_all, y_all, a_all):
            y = int(y)
            a = min(a, max_a)
            a = max(a, min_a)
            c = int(self.output_nodes * (a - min_a) / (max_a - min_a))
            if c == self.output_nodes:
                c = self.output_nodes - 1
            data[y][c].append(x)
        return data

    def _merge(self, y, y2, samples1, samples2):
        return random.choice(samples1[y][y2])

    def compute_input_shape(self):
        return self.compute_one_input_shape()

    def compute_one_input_shape(self):
        return self.shape1


class LengthDataGenerator(TextDataGenerator):
    def __init__(self, args):
        self.args = args
        if args.dataset1 == 'imdb':
            self.output_nodes = 2
        else:
            self.output_nodes = 10

        self.maxlen = 200
        self.max_length = self.maxlen + 1
        train1, test1, self.shape1 = self._get_data(args.dataset1, self.maxlen)
        train2, test2, self.shape2 = train1, test1, self.shape1
        self.input_shape = self.compute_input_shape()

        inputs = [train1[0], test1[0]]
        self.vocab_size = max(max([max(x) for x in xs]) for xs in inputs) + 1

        train = [np.concatenate([train1[i], test1[i]], 0) for i in range(2)]
        # Preprocessing
        self.train_samples1 = self._prepare_data(train)
        self.test_samples1 = self.train_samples1
        self.train_samples2 = self.train_samples1
        self.test_samples2 = self.test_samples1

        self.train_label_pairs = []
        self.test_label_pairs = []
        self.get_label_splits()

    def _prepare_data(self, data):
        x_all, y_all = data
        assert len(x_all) == len(y_all)

        data = [[[] for _ in range(self.output_nodes)] for _ in
                range(self.output_nodes)]
        for x, y in zip(x_all, y_all):
            y = int(y)
            l = len(x)
            if l < 50:
                a = 0
            else:
                a = (l - 50) // 15
            padded_x = np.array(x + ([0] * (self.max_length - len(x))))
            data[y][a].append(padded_x)
        return data

    def _merge(self, y, y2, samples1, samples2):
        return random.choice(samples1[y][y2])

    def _get_random_input(self):
        length = np.random.randint(1, high=self.maxlen)
        x = [1] + np.random.randint(2, high=self.vocab_size,
                                    size=length).tolist()
        padding = [0] * (self.max_length - len(x))
        x = x + padding
        return x
