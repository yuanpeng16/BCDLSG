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
        ev = FocusedPartitionEvaluator(args, model, datasets, large_datasets,
                                       test_label_pairs)
    elif args.evaluator_type == 'output':
        ev = OutputEvaluator(args, model, datasets, large_datasets,
                             test_label_pairs)
    elif args.evaluator_type == 'filtered':
        ev = FilteredOutputEvaluator(args, model, datasets, large_datasets,
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

    def filter(self, x, f):
        return {key: value for key, value in x.items() if key in f}

    def get_perplexity(self, counts):
        if len(counts) == 0:
            return 0.0
        count_list = [value for _, value in counts.items()]
        entropy = get_entropy(count_list)
        perplexity = 2 ** entropy
        return perplexity

    def evaluate_partitions(self, train_prediction, random_prediction):
        dense_train_partition = set(self.get_elements(train_prediction))
        random_elements = self.get_elements(random_prediction)

        # partitions
        random_partition = set(random_elements)
        test_partition = random_partition.difference(dense_train_partition)
        train_partition = random_partition.difference(test_partition)

        random_num = len(random_partition)
        train_num = len(train_partition)
        test_num = len(test_partition)
        partitions = [random_num, train_num, test_num]

        # entropy
        random_count = self.get_counts(random_elements)
        train_count = self.filter(random_count, train_partition)
        test_count = self.filter(random_count, test_partition)

        random_entropy = self.get_perplexity(random_count)
        train_entropy = self.get_perplexity(train_count)
        test_entropy = self.get_perplexity(test_count)
        entropies = [random_entropy, train_entropy, test_entropy]

        ret = partitions + entropies
        return ret

    def evaluate_datasets(self, datasets):
        datasets = self.large_datasets
        assert len(datasets) == 3
        train_dataset, _, random_dataset = datasets
        train_prediction = self.forward(train_dataset[0])
        all_prediction = self.forward(random_dataset[0])

        train_acc = self.get_accuracy(train_prediction, train_dataset[1])
        ret = self.evaluate_partitions(train_prediction, all_prediction)
        return ret + [100 * train_acc[2]]


class FocusedPartitionEvaluator(PartitionEvaluator):
    def evaluate_partitions(self, train_prediction, random_prediction):
        random_elements = self.get_elements(random_prediction)
        all_train_elements = self.get_elements(train_prediction)

        # partitions
        random_partition = set(random_elements)
        all_train_partition = set(all_train_elements)
        train_partition = random_partition.intersection(all_train_partition)
        test_partition = random_partition.difference(all_train_partition)

        random_num = len(random_partition)
        train_num = len(train_partition)
        test_num = len(test_partition)
        partitions = [random_num, train_num, test_num]

        # counts
        random_count = self.get_counts(random_elements)
        label_count = self.filter(random_count, self.test_label_pairs)
        test_count = self.filter(random_count, test_partition)

        label_samples = sum([v for _, v in label_count.items()])
        test_samples = sum([v for _, v in test_count.items()])
        counts = [(100 * test_samples) / len(random_elements),
                  (100 * label_samples) / len(random_elements)]

        ret = partitions + counts
        return ret

    def evaluate_datasets(self, datasets):
        datasets = self.large_datasets
        assert len(datasets) == 3
        train_dataset, test_dataset, _ = datasets
        train_prediction = self.forward(train_dataset[0])
        test_prediction = self.forward(test_dataset[0])

        train_acc = self.get_accuracy(train_prediction, train_dataset[1])
        ret = self.evaluate_partitions(train_prediction, test_prediction)
        return ret + [100 * train_acc[2]]


class OutputEvaluator(Evaluator):
    def output_evaluate(self, x, y, output_labels):
        y_hat = self.forward(x)
        n_samples = len(y[0])
        y1_hat_list = y_hat[0]
        y2_hat_list = y_hat[1]
        hit1, hit2, hit, sg_hit = 0, 0, 0, 0
        for i in range(n_samples):
            y1_hat = y1_hat_list[i]
            y2_hat = y2_hat_list[i]
            if (y1_hat, y2_hat) not in output_labels:
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

    def get_output_labels(self, y1_hat_list, y2_hat_list):
        output_labels = set()
        for y1_hat, y2_hat in zip(y1_hat_list, y2_hat_list):
            output_labels.add((y1_hat, y2_hat))
        return output_labels

    def evaluate_datasets(self, datasets):
        x, y = self.large_datasets[0]
        y_hat = self.forward(x)
        y1_hat_list = y_hat[0]
        y2_hat_list = y_hat[1]
        output_labels = self.get_output_labels(y1_hat_list, y2_hat_list)

        ret = []
        for dataset in datasets:
            ret.extend(
                self.output_evaluate(dataset[0], dataset[1], output_labels))
            ret.append("\t")
        return ret


class FilteredOutputEvaluator(OutputEvaluator):
    def get_output_labels(self, y1_hat_list, y2_hat_list):
        ret = super().get_output_labels(y1_hat_list, y2_hat_list)
        print('Outputs', len(ret))
        return ret

    def get_output_labels1(self, y1_hat_list, y2_hat_list):
        output_labels = {}
        for y1_hat, y2_hat in zip(y1_hat_list, y2_hat_list):
            key = (y1_hat, y2_hat)
            output_labels[key] = output_labels.get(key, 0) + 1
        terms = list(output_labels.items())
        print(len(terms))
        print(terms[:5])
        terms = sorted(terms, key=lambda x: x[1], reverse=True)
        print(terms[:5])
        exit()
        return output_labels


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
