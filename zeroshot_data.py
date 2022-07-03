import scipy.io
import numpy as np


def get_feat(fn):
    mat = scipy.io.loadmat(fn)
    feat = mat['feat']
    matrix = feat.toarray()
    matrix = np.transpose(matrix)
    data = np.reshape(matrix, [-1, 199, 7, 7])
    data = np.transpose(data, [0, 2, 3, 1])
    return data


def get_label(weights, x):
    return sum([a * b for a, b in zip(weights, x)])


def get_combined_labels(matrix):
    mean = np.mean(matrix, 0)
    distance = np.abs(mean - 0.5)
    ordered = sorted(enumerate(distance), key=lambda x: x[1])
    first_labels = [ordered[0][0], ordered[2][0], ordered[4][0]]
    second_labels = [ordered[1][0], ordered[3][0], ordered[5][0]]
    return first_labels, second_labels


def get_label_matrix(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    matrix = [[int(x) for x in line.strip().split()[6:]] for line in lines]
    matrix = np.asarray(matrix)
    return matrix


def get_labels(matrix, combined_labels):
    first_labels, second_labels = combined_labels
    assert len(first_labels) == len(second_labels)
    weights = [2 ** i for i in range(len(first_labels))]

    outputs = []
    for labels in matrix:
        first_output = get_label(weights, [labels[i] for i in first_labels])
        second_output = get_label(weights, [labels[i] for i in second_labels])
        outputs.append((first_output, second_output))
    return outputs


def one_hot(a, output_nodes):
    ret = [0] * output_nodes
    ret[a] = 1
    return ret


def one_hot_pair(y, output_nodes):
    return [one_hot(a, output_nodes) for a in y]


class ZeroShotDataGenerator(object):
    def __init__(self, args):
        self.args = args
        self.input_shape = (7, 7, 199)
        self.output_nodes = 2 ** 3

        # prepare label combinations
        self.test_label_pairs = set()
        for x in range(self.output_nodes):
            for y in range(self.output_nodes):
                if not self.is_train_label(x, y):
                    self.test_label_pairs.add((x, y))

        # load data
        path = '/home/yuanpeng/datasets/apy/'
        x_folder = path + 'attribute_features/'
        y_folder = path + 'attribute_data/'
        fn_x_train = x_folder + 'feat_apascal_train.mat'
        fn_y_train = y_folder + 'apascal_train.txt'
        fn_x_test = x_folder + 'feat_apascal_test.mat'
        fn_y_test = y_folder + 'apascal_test.txt'
        train_label_matrix = get_label_matrix(fn_y_train)
        test_label_matrix = get_label_matrix(fn_y_test)
        combined_labels = get_combined_labels(train_label_matrix)
        train_labels = get_labels(train_label_matrix, combined_labels)
        test_labels = get_labels(test_label_matrix, combined_labels)
        self.train_samples = self.load_data(fn_x_train, train_labels, True)
        self.test_samples = self.load_data(fn_x_test, test_labels, False)

    def load_data(self, fn_x, labels, is_train):
        feats = get_feat(fn_x)
        assert len(feats) == len(labels)
        x_list = []
        y_list = []
        for x, y in zip(feats, labels):
            if is_train:
                if y not in self.test_label_pairs:
                    x_list.append(x)
                    y_list.append(one_hot_pair(y, self.output_nodes))
            else:
                if y in self.test_label_pairs:
                    x_list.append(x)
                    y_list.append(one_hot_pair(y, self.output_nodes))
        return np.asarray(x_list), np.asarray(y_list)

    def is_train_label(self, x, y):
        diff = (y - x + self.output_nodes) % self.output_nodes
        return diff < self.output_nodes // 2

    def get_input_shape(self):
        return self.input_shape

    def get_output_nodes(self):
        return self.output_nodes

    def get_test_label_pairs(self):
        return self.test_label_pairs

    def get_vocab_size(self):
        return 0

    def _get_samples(self, data, k, pretrain=False):
        n_samples = len(data[0])
        index_list = np.random.choice(n_samples, size=k)
        x = np.asarray([data[0][i] for i in index_list])
        y = np.asarray([data[1][i] for i in index_list])
        y = np.transpose(y, [1, 0, 2])
        y1, y2 = y[0], y[1]
        return [x, [y1, y2]]

    def get_training_samples(self, k, pretrain=False):
        return self._get_samples(self.train_samples, k, pretrain=pretrain)

    def get_eval_samples(self, k):
        return self.get_training_samples(k)

    def get_test_samples(self, k, randomize=False):
        samples, y_list = self._get_samples(self.test_samples, k)
        if randomize:
            samples = np.random.rand(*samples.shape)
        return samples, y_list


if __name__ == '__main__':
    dg = ZeroShotDataGenerator(None)
    print(dg.get_training_samples(3))
    print(dg.get_test_samples(3))
