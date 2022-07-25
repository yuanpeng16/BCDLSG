import scipy.io
import numpy as np
import os


def get_label(weights, x):
    return sum([a * b for a, b in zip(weights, x)])


def one_hot(a, output_nodes):
    ret = [0] * output_nodes
    ret[a] = 1
    return ret


def one_hot_pair(y, output_nodes):
    return [one_hot(a, output_nodes) for a in y]


class ZeroShotDataGenerator(object):
    def __init__(self, args):
        self.args = args
        self.labels = args.combined_labels
        self.dataset_dir = args.dataset_dir
        self.output_nodes = 2 ** self.labels

        # prepare label combinations
        self.test_label_pairs = set()
        for x in range(self.output_nodes):
            for y in range(self.output_nodes):
                if not self.is_train_label(x, y):
                    self.test_label_pairs.add((x, y))
        # load data
        self.train_samples, self.test_samples = self.get_data()
        self.input_shape = self.train_samples[0][0].shape

    def get_data(self):
        raise NotImplementedError()

    def get_combined_labels(self, matrix):
        mean = np.mean(matrix, 0)
        distance = np.abs(mean - 0.5)
        ordered = sorted(enumerate(distance), key=lambda x: x[1])
        first_labels = [ordered[2 * i][0] for i in range(self.labels)]
        second_labels = [ordered[2 * i + 1][0] for i in range(self.labels)]
        return first_labels, second_labels

    def get_labels(self, matrix, combined_labels):
        first_labels, second_labels = combined_labels
        assert len(first_labels) == len(second_labels)
        weights = [2 ** i for i in range(len(first_labels))]

        outputs = []
        for labels in matrix:
            first_output = get_label(weights,
                                     [labels[i] for i in first_labels])
            second_output = get_label(weights,
                                      [labels[i] for i in second_labels])
            outputs.append((first_output, second_output))
        return outputs

    def is_train_label(self, x, y):
        half = self.output_nodes // 2
        if self.args.label_split == 'tile':
            return x < half or y < half
        elif self.args.label_split == 'diagonal':
            diff = (y - x + self.output_nodes) % self.output_nodes
            return diff < half
        else:
            raise ValueError('{0} is not a valid label_split.'.format(
                self.args.label_split))

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

    def split_data(self, feats, labels):
        assert len(feats) == len(labels)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for x, y in zip(feats, labels):
            y_one_hot = one_hot_pair(y, self.output_nodes)
            if y not in self.test_label_pairs:
                x_train.append(x)
                y_train.append(y_one_hot)
            else:
                x_test.append(x)
                y_test.append(y_one_hot)
        train_data = [np.asarray(x_train), np.asarray(y_train)]
        test_data = [np.asarray(x_test), np.asarray(y_test)]
        return train_data, test_data


class APYDataGenerator(ZeroShotDataGenerator):
    def get_data(self):
        path = os.path.join(self.dataset_dir, 'apy')
        x_folder = os.path.join(path, 'attribute_features')
        y_folder = os.path.join(path, 'attribute_data')
        fn_x_train = os.path.join(x_folder, 'feat_apascal_train.mat')
        fn_y_train = os.path.join(y_folder, 'apascal_train.txt')
        fn_x_test = os.path.join(x_folder, 'feat_apascal_test.mat')
        fn_y_test = os.path.join(y_folder, 'apascal_test.txt')
        train_label_matrix = self.get_label_matrix(fn_y_train)
        test_label_matrix = self.get_label_matrix(fn_y_test)
        label_matrix = np.concatenate(
            [train_label_matrix, test_label_matrix], 0)
        combined_labels = self.get_combined_labels(label_matrix)
        train_labels = self.get_labels(train_label_matrix, combined_labels)
        test_labels = self.get_labels(test_label_matrix, combined_labels)
        train_samples = self.load_data(fn_x_train, train_labels, True)
        test_samples = self.load_data(fn_x_test, test_labels, False)
        return train_samples, test_samples

    def get_feat(self, fn):
        mat = scipy.io.loadmat(fn)
        feat = mat['feat']
        matrix = feat.toarray()
        matrix = np.transpose(matrix)
        data = np.reshape(matrix, [-1, 7, 7, 199])
        return data

    def get_label_matrix(self, fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
        matrix = [[int(x) for x in line.strip().split()[6:]] for line in lines]
        matrix = np.asarray(matrix)
        return matrix

    def load_data(self, fn_x, labels, is_train):
        feats = self.get_feat(fn_x)
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


class AWA2DataGenerator(ZeroShotDataGenerator):
    def get_data(self):
        path = os.path.join(self.dataset_dir, 'awa2')
        x_folder = os.path.join(
            path, 'AwA2-features/Animals_with_Attributes2/Features/ResNet101')
        y_folder = os.path.join(path, 'AwA2-base/Animals_with_Attributes2')
        fn_x_train = os.path.join(x_folder, 'AwA2-features.txt')
        fn_z_train = os.path.join(x_folder, 'AwA2-labels.txt')
        fn_y_train = os.path.join(y_folder, 'predicate-matrix-binary.txt')
        return self.load_data(fn_x_train, fn_z_train, fn_y_train)

    def get_feat(self, fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
        matrix = [[float(x) for x in line.strip().split()] for line in lines]
        matrix = np.asarray(matrix)
        data = np.reshape(matrix, [-1, 8, 8, 32])
        # data = np.reshape(matrix, [-1, 32, 8, 8])
        # data = np.transpose(data, [0, 2, 3, 1])
        return data

    def load_labels(self, fn, afn):
        with open(fn, 'r') as f:
            lines = f.readlines()
        labels = [int(line.strip()) for line in lines]

        with open(afn, 'r') as f:
            lines = f.readlines()
        attributes = [[int(x) for x in line.strip().split()] for line in lines]

        matrix = [attributes[y - 1] for y in labels]
        matrix = np.asarray(matrix)
        combined_labels = self.get_combined_labels(matrix)
        return self.get_labels(matrix, combined_labels)

    def load_data(self, fn_x, fn_z, fn_y):
        labels = self.load_labels(fn_z, fn_y)
        feats = self.get_feat(fn_x)
        return self.split_data(feats, labels)


class PreprocessedDataGenerator(ZeroShotDataGenerator):
    def load_data(self, fn_f_train, fn_y):
        labels = self.load_labels(fn_y)
        feats = np.load(fn_f_train, allow_pickle=True)
        feats = feats / 255.0
        ret = self.split_data(feats, labels)
        return ret

    def load_labels(self, fn):
        raise NotImplementedError()


class CUBDataGenerator(PreprocessedDataGenerator):
    def get_data(self):
        path = os.path.join(self.dataset_dir,
                            'cub/CUB2002011/CUB_200_2011/CUB_200_2011')
        y_folder = os.path.join(path, 'attributes')
        fn_y_train = os.path.join(y_folder, 'image_attribute_labels.txt')
        fn_f_train = os.path.join(path, 'feat.npy')
        return self.load_data(fn_f_train, fn_y_train)

    def load_labels(self, fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
        attributes = [int(line.strip().split()[2]) for line in lines]
        matrix = np.asarray(attributes)
        matrix = np.reshape(matrix, [-1, 312])
        combined_labels = self.get_combined_labels(matrix)
        return self.get_labels(matrix, combined_labels)


class SUNDataGenerator(PreprocessedDataGenerator):
    def get_data(self):
        path = os.path.join(self.dataset_dir, 'sun')
        y_folder = os.path.join(path, 'SUNAttributeDB')
        fn_y_train = os.path.join(y_folder, 'attributeLabels_continuous.mat')
        fn_f_train = os.path.join(path, 'feat.npy')
        return self.load_data(fn_f_train, fn_y_train)

    def load_labels(self, fn):
        mat = scipy.io.loadmat(fn)
        feat = mat['labels_cv']
        matrix = np.round(feat).astype(int)
        combined_labels = self.get_combined_labels(matrix)
        return self.get_labels(matrix, combined_labels)


if __name__ == '__main__':
    dg = SUNDataGenerator(None)
    print(dg.get_training_samples(3))
    print(dg.get_test_samples(3))
