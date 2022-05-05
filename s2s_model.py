import tensorflow as tf
import numpy as np
import random


def clip_gradient_norms(gradients, max_norm):
    clipped_gradients = []
    for grad in gradients:
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                tmp = tf.clip_by_norm(grad.values, max_norm)
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad = tf.clip_by_norm(grad, max_norm)
        clipped_gradients.append(grad)
    return clipped_gradients


class SimpleModel(object):
    def __init__(self, args):
        self.args = args
        if args.random_random:
            tf.set_random_seed(random.randint(2, 1000))
        else:
            tf.set_random_seed(args.random_seed)
        self.max_input_length = args.input_length
        self.max_output_length = args.output_length

    def get_optimizer(self, scope=None):
        if self.args.decay_steps <= 0:
            learning_rate = self.args.learning_rate
        else:
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.args.learning_rate
            decay_steps = self.args.decay_steps
            decay_base = 0.96
            learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                       global_step,
                                                       decay_steps, decay_base,
                                                       staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        if self.args.max_gradient_norm < 0:
            if self.args.decay_steps <= 0:
                optimizer_op = optimizer.minimize(self.loss)
            else:
                optimizer_op = optimizer.minimize(self.loss,
                                                  global_step=global_step)
        else:
            params = tf.trainable_variables(scope)
            gradients = tf.gradients(self.loss, params)
            if self.args.clip_by_norm:
                clipped_gradients = clip_gradient_norms(gradients,
                                                        self.args.max_gradient_norm)
            else:
                clipped_gradients, _ = tf.clip_by_global_norm(
                    gradients, self.args.max_gradient_norm)

            if self.args.decay_steps <= 0:
                optimizer_op = optimizer.apply_gradients(
                    zip(clipped_gradients, params))
            else:
                optimizer_op = optimizer.apply_gradients(
                    zip(clipped_gradients, params), global_step=global_step)
        return optimizer_op

    def initialize(self, voc_size, act_size):
        self.V = voc_size
        self.U = act_size
        self.create_model()

        self.optimizer_op = self.get_optimizer()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()

        tf.summary.FileWriter('logs/' + self.args.experiment_id + '/model',
                              self.sess.graph)
        self.sess.run(init)

    def random_select(self, sample_list, batch_size):
        length = len(sample_list[0])
        indice = np.random.choice(length, size=batch_size)
        result = [[samples[i] for i in indice] for samples in sample_list]
        return result

    def train(self, train_data, val_data, test_data):
        _, sent_acc_val, loss_val = self.test(*val_data)
        _, sent_acc_test, loss_test = self.test(*test_data)
        print(0, 0, 0, loss_val, sent_acc_val, loss_test, sent_acc_test)

        log_step = 1
        fetch = [self.optimizer_op, self.loss, self.word_accuracy,
                 self.sent_accuracy]

        avg_loss, avg_wa, avg_sa = 0, 0, 0
        for i in range(self.args.epochs):
            batch = self.random_select(train_data, self.args.batch_size)
            feed = {self.x: batch[0], self.y: batch[1], self.x_len: batch[2],
                    self.y_len: batch[3]}
            _, loss, word_acc, sent_acc = self.sess.run(fetch, feed_dict=feed)
            avg_loss += loss
            avg_wa += word_acc
            avg_sa += sent_acc
            if i % log_step == 0:
                _, sent_acc_val, loss_val = self.test(*val_data)
                _, sent_acc_test, loss_test = self.test(*test_data)
                train_loss = avg_loss / log_step
                train_acc = avg_sa / log_step
                print(i + 1, train_loss, train_acc, loss_val, sent_acc_val,
                      loss_test, sent_acc_test)
                avg_loss, avg_wa, avg_sa = 0, 0, 0

    def test(self, X, Y, X_len, Y_len):
        fetch = [self.loss, self.word_accuracy, self.sent_accuracy,
                 self.prediction]
        feed = {self.x: X, self.y: Y, self.x_len: X_len, self.y_len: Y_len}
        loss, word_scc, sent_acc, prediction = self.sess.run(
            fetch, feed_dict=feed)
        return prediction, sent_acc, loss


class S2SModel(SimpleModel):
    # Encoder
    def get_embeddings(self, inputs, embedding_size):
        embeddings = tf.Variable(
            tf.random_uniform([self.V, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, inputs)
        return embed

    def get_encoder_bidirectional(self, x):
        encoder_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.args.num_units / 2)
        encoder_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.args.num_units / 2)
        encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
            encoder_cell_fw, encoder_cell_bw, x, sequence_length=self.x_len,
            dtype=tf.float32)
        h = tf.concat([encoder_state[0].h, encoder_state[1].h], 1)
        c = tf.concat([encoder_state[0].c, encoder_state[1].c], 1)
        state = tf.nn.rnn_cell.LSTMStateTuple(h, c)
        return tf.concat(encoder_outputs, 2), state

    def get_encoder(self, x):
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.args.num_units)
        return tf.nn.dynamic_rnn(encoder_cell, x, sequence_length=self.x_len,
                                 dtype=tf.float32)

    # Decoder
    def get_decoder_input(self):
        if not self.args.use_decoder_input:
            zeros = tf.zeros(shape=self.batch_size)
            unsqueezed = tf.reshape(zeros, [-1, 1, 1])
            decoder_emb_inp = tf.tile(unsqueezed,
                                      [1, self.max_output_length, 1])
            return decoder_emb_inp
        else:
            embeddings = tf.Variable(
                tf.random_uniform([self.U, self.args.output_embedding_size],
                                  -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, self.y)
            decoder_emb_inp = tf.manip.roll(embed, 1, 1)
            return decoder_emb_inp

    def get_decoder_cell(self, encoder_state, encoder_outputs, decoder_cell,
                         num_units, source_sequence_length):
        attention_states = encoder_outputs
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, attention_states,
            memory_sequence_length=source_sequence_length)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=num_units)

        initial_state = decoder_cell.zero_state(dtype=tf.float32,
                                                batch_size=self.batch_size)
        initial_state = initial_state.clone(cell_state=encoder_state)
        return decoder_cell, initial_state

    def get_logits(self, x):
        num_units = self.args.num_units
        source_sequence_length = self.x_len

        # Encoder
        if self.args.bidirectional_encoder:
            encoder_outputs, encoder_state = self.get_encoder_bidirectional(x)
        else:
            encoder_outputs, encoder_state = self.get_encoder(x)

        # Decoder
        decoder_emb_inp = self.get_decoder_input()
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        decoder_cell, encoder_state = self.get_decoder_cell(
            encoder_state, encoder_outputs, decoder_cell,
            num_units, source_sequence_length)

        ones = tf.ones(shape=self.batch_size, dtype=tf.int32)
        lengths = ones * self.max_output_length
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                                  encoder_state)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        score = tf.layers.dense(outputs.rnn_output, self.U)
        return score

    def create_model(self):
        self.x = tf.placeholder(tf.int64, shape=(None, self.max_input_length,),
                                name='x')
        self.y = tf.placeholder(tf.int64,
                                shape=(None, self.max_output_length,),
                                name='y')
        self.x_len = tf.placeholder(tf.int32, shape=(None,), name='x_len')
        self.y_len = tf.placeholder(tf.int32, shape=(None,), name='y_len')
        self.batch_size = tf.shape(self.x_len)

        # masks
        self.target_mask_float = tf.sequence_mask(
            self.y_len, maxlen=self.max_output_length, dtype=tf.float32)
        self.target_mask_int64 = tf.sequence_mask(
            self.y_len, maxlen=self.max_output_length, dtype=tf.int64)

        with tf.variable_scope("word_embeddings"):
            embedding = self.get_embeddings(self.x, self.args.embedding_size)

        with tf.variable_scope("generate_logits"):
            l = self.get_logits(embedding)

        with tf.variable_scope("evaluation"):
            # loss
            if self.args.hinge_loss:
                loss_sum = tf.losses.hinge_loss(
                    labels=tf.one_hot(self.y, self.U), logits=l)
            else:
                loss_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y, logits=l)
            self.loss = tf.reduce_sum(
                loss_sum * self.target_mask_float / tf.to_float(
                    self.batch_size))

            # word accuracy
            self.prediction = tf.argmax(l, -1) * self.target_mask_int64
            word_equality = tf.to_float(tf.equal(self.y, self.prediction))
            valid_word_equality = word_equality * self.target_mask_float
            self.word_accuracy = tf.reduce_mean(tf.reduce_sum(
                valid_word_equality, -1) / (tf.to_float(self.y_len)))

            # sentence accuracy
            sent_equality = tf.reduce_min(word_equality, axis=-1)
            self.sent_accuracy = tf.reduce_mean(sent_equality)
