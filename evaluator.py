import tensorflow as tf
import numpy as np

from data import one_hot


def get_entropy(counts, normalize=True):
    if len(counts) == 0:
        return float('-inf')
    assert min(counts) >= 0
    assert sum(counts) > 0
    counts = [a for a in counts if a > 0]
    x = np.asarray(counts)
    assert len(x.shape) == 1
    if normalize:
        den = np.sum(x)
        x = x / den
    log_x = np.log2(x)
    entropy = -np.dot(log_x, x)
    return entropy


def get_perplexity(counts):
    if len(counts) == 0:
        return 0.0
    count_list = [value for _, value in counts.items()]
    entropy = get_entropy(count_list)
    perplexity = 2 ** entropy
    return perplexity


def get_evaluator(args, model, datasets, large_datasets, test_label_pairs):
    if args.adversarial:
        if args.any_generalization:
            ev = RandomAdversarialEvaluator(
                args, model, datasets, large_datasets, test_label_pairs)
        else:
            ev = AdversarialEvaluator(
                args, model, datasets, large_datasets, test_label_pairs)
    elif args.evaluator_type == 'partition':
        ev = PartitionEvaluator(args, model, datasets, large_datasets,
                                test_label_pairs)
    elif args.evaluator_type == 'partition-f':
        ev = FilteredPartitionEvaluator(args, model, datasets, large_datasets,
                                        test_label_pairs)
    elif args.evaluator_type == 'partition-t':
        ev = ThresholdPartitionEvaluator(args, model, datasets, large_datasets,
                                         test_label_pairs)
    elif args.evaluator_type == 'single':
        ev = SingleEvaluator(args, model, datasets, large_datasets,
                               test_label_pairs)
    else:
        ev = Evaluator(args, model, datasets, large_datasets, test_label_pairs)
    return ev


class Evaluator(object):
    def __init__(self, args, model, datasets, large_datasets,
                 test_label_pairs):
        self.args = args
        self.model = model
        self.datasets = datasets
        self.large_datasets = large_datasets
        self.test_label_pairs = set(test_label_pairs)

    def forward(self, x):
        y1_hat, y2_hat = [], []
        size = self.args.batch_size
        for i in range(0, len(x), size):
            j = min(i + size, len(x))
            y1, y2 = self.model(x[i:j])
            y1 = tf.argmax(y1, -1).numpy()
            y2 = tf.argmax(y2, -1).numpy()
            y1_hat.extend(y1)
            y2_hat.extend(y2)
        return y1_hat, y2_hat

    def get_accuracy(self, y_hat, y):
        n_samples = len(y[0])
        y1_hat_list = y_hat[0]
        y2_hat_list = y_hat[1]
        hit1, hit2, hit, sg_hit = 0, 0, 0, 0
        for i in range(n_samples):
            y1_hat = y1_hat_list[i]
            y2_hat = y2_hat_list[i]
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

    def evaluate(self, x, y):
        y_hat = self.forward(x)
        return self.get_accuracy(y_hat, y)

    def test_evaluate(self, x, y):
        return self.evaluate(x, y)

    def evaluate_datasets(self, datasets):
        ret = []
        for dataset in datasets:
            ret.extend(
                self.evaluate(dataset[0], dataset[1]))
            ret.append("\t")
        return ret

    def evaluate_all(self):
        return self.evaluate_datasets(self.datasets)

    def large_evaluate_all(self):
        return self.evaluate_datasets(self.large_datasets)


class PartitionEvaluator(Evaluator):
    def __init__(self, args, model, datasets, large_datasets,
                 test_label_pairs):
        super().__init__(args, model, datasets, large_datasets,
                         test_label_pairs)

        y = self.large_datasets[0][1]
        n_first_factor = len(y[0][0])
        n_second_factor = len(y[1][0])
        self.n_possible_outputs = n_first_factor * n_second_factor

    def evaluate_all(self):
        return self.large_evaluate_all()

    def get_counts(self, elements):
        counts = {}
        for y1_hat, y2_hat in elements:
            key = (y1_hat, y2_hat)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_elements(self, y_hat):
        y1_hat_list = y_hat[0]
        y2_hat_list = y_hat[1]
        return [(a, b) for a, b in zip(y1_hat_list, y2_hat_list)]

    def split_test_data(self, train_labels, test_counts, opposite=False):
        test_in_train = {}
        test_in_test = {}
        for key, value in test_counts.items():
            if key in train_labels:
                test_in_train[key] = value
            else:
                test_in_test[key] = value

        if opposite:
            test_in_test, test_in_train = test_in_train, test_in_test
        return test_in_train, test_in_test

    def get_values(self, train_labels, test_counts, all_test_counts,
                   opposite=False):
        # Partitions
        test_in_train, test_in_test = self.split_test_data(
            train_labels, test_counts, opposite=opposite)

        # Ratio
        _, all_test_in_test = self.split_test_data(
            train_labels, all_test_counts, opposite=opposite)
        test_num = sum([v for _, v in all_test_counts.items()])
        predict_num = sum([v for _, v in all_test_in_test.items()])
        ratio = (100 * predict_num) / test_num

        predict = [len(test_in_train), len(test_in_test), ratio]
        return predict

    def filter_counts(self, counts):
        return counts

    def evaluate_partitions(self, train_labels, test_prediction):
        all_test_elements = self.get_elements(test_prediction)
        all_test_counts = self.get_counts(all_test_elements)
        test_counts = self.filter_counts(all_test_counts)

        predict = self.get_values(train_labels, test_counts, all_test_counts)
        truth = self.get_values(self.test_label_pairs, test_counts,
                                all_test_counts, opposite=True)
        return predict + truth

    def evaluate_datasets(self, datasets):
        """
        Return statistics
        :param datasets:
        :return:
        01 step
        02 # of prediction i.i.d. test partitions
        03 # of prediction o.o.d. test partitions
        04 Ratio of prediction o.o.d. test samples
        05 # of ground-truth i.i.d. test partitions
        06 # of ground-truth o.o.d. test partitions
        07 Ratio of ground-truth o.o.d. test samples
        08 # of prediction i.i.d. random partitions
        09 # of prediction o.o.d. random partitions
        10 Ratio of prediction o.o.d. random samples
        11 # of ground-truth i.i.d. random partitions
        12 # of ground-truth o.o.d. random partitions
        13 Ratio of ground-truth o.o.d. random samples
        14 # of training partitions
        15 Training sample accuracy
        """
        assert len(datasets) == 3
        train_dataset, test_dataset, random_dataset = datasets
        train_prediction = self.forward(train_dataset[0])
        test_prediction = self.forward(test_dataset[0])
        random_prediction = self.forward(random_dataset[0])

        train_acc = self.get_accuracy(train_prediction, train_dataset[1])

        # train partitions
        all_train_elements = self.get_elements(train_prediction)
        all_train_counts = self.get_counts(all_train_elements)
        train_counts = self.filter_counts(all_train_counts)
        train_labels = set([k for k, _ in train_counts.items()])

        test_ret = self.evaluate_partitions(train_labels, test_prediction)
        random_ret = self.evaluate_partitions(train_labels, random_prediction)
        return test_ret + random_ret + [len(train_labels), 100 * train_acc[2]]


class FilteredPartitionEvaluator(PartitionEvaluator):
    def filter_counts(self, counts):
        items = list(counts.items())
        items = sorted(items, key=lambda x: x[1], reverse=True)
        size = min(len(items), len(self.test_label_pairs))
        filtered = items[:size]
        ret = {k: v for (k, v) in filtered}
        return ret


class ThresholdPartitionEvaluator(PartitionEvaluator):
    def filter_counts(self, counts):
        n_all_samples = sum([v for _, v in counts.items()])
        nom = self.args.partition_threshold_percentage * n_all_samples
        den = 100 * self.n_possible_outputs
        ret = {k: v for k, v in counts.items() if den * v >= nom}
        return ret


class SingleEvaluator(Evaluator):
    def forward(self, x):
        y1_hat = []
        size = self.args.batch_size
        for i in range(0, len(x), size):
            j = min(i + size, len(x))
            y1 = self.model(x[i:j])
            y1 = tf.argmax(y1, -1).numpy()
            y1_hat.extend(y1)
        return y1_hat

    def get_accuracy(self, y_hat, y):
        n_samples = len(y)
        y1_hat_list = y_hat
        hit1, sg_hit = 0, 0
        for i in range(n_samples):
            y1_hat = y1_hat_list[i]
            if (y1_hat, y1_hat) in self.test_label_pairs:
                sg_hit += 1
            h1 = y[i][y1_hat] == 1
            if h1:
                hit1 += 1
        acc1 = hit1 / n_samples
        sg_acc = sg_hit / n_samples
        return 0, 0, acc1, sg_acc


class AdversarialEvaluator(Evaluator):
    def __init__(self, args, model, datasets, large_datasets,
                 test_label_pairs):
        super().__init__(args, model, datasets, large_datasets,
                         test_label_pairs)
        self.initialize()

    def initialize(self):
        self.loss_object_list = [tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction='none') for _ in range(2)]

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
                from_logits=False, reduction='none') for _ in
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
