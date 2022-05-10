import tensorflow as tf
import numpy as np

from mnist_data import one_hot


def get_evaluator(args, model, datasets, large_datasets, test_label_pairs):
    if args.adversarial:
        if args.any_generalization:
            ev = RandomAdversarialEvaluator(
                args, model, datasets, large_datasets, test_label_pairs)
        else:
            ev = AdversarialEvaluator(
                args, model, datasets, large_datasets, test_label_pairs)
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
            y1_hat.extend(y1)
            y2_hat.extend(y2)
        return y1_hat, y2_hat

    def evaluate(self, x, y):
        y_hat = self.forward(x)
        n_samples = len(y[0])
        y1_hat_list = np.argmax(y_hat[0], -1)
        y2_hat_list = np.argmax(y_hat[1], -1)
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

    def evaluate2(self, x, y):
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
