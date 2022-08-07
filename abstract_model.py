import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense


class AbstractModelGenerator(object):
    def __init__(self, args, input_shape, output_nodes):
        self.args = args
        self.input_shape = input_shape
        self.output_nodes = output_nodes
        self.depth = self.args.n_common_layers + self.args.n_separate_layers
        self.vocab_size = 0

    def convert_input(self, x):
        return x

    def get_intermediate_layer(self, hn, x):
        return x

    def get_first_layer(self, hn, x):
        return self.get_intermediate_layer(hn, x)

    def get_last_layer(self, hn, x):
        return self.get_intermediate_layer(hn, x)

    def get_one_layer(self, hn, x, index, part):
        if index == 0:
            x = self.get_first_layer(hn, x)
        elif index == self.depth - 1:
            x = self.get_last_layer(hn, x)
        else:
            x = self.get_intermediate_layer(hn, x)
        return x

    def get_output_layer(self, x, activation, name):
        return Dense(self.output_nodes, activation=activation, name=name)(x)

    def constant_n_hidden_nodes(self):
        return False

    def get_main_model(self, x):
        x = self.convert_input(x)
        for i in range(self.args.n_common_layers):
            x = self.get_one_layer(self.args.n_hidden_nodes, x, i, 0)
        x1, x2 = x, x

        if self.constant_n_hidden_nodes():
            h1, h2 = self.args.n_hidden_nodes, self.args.n_hidden_nodes
        else:
            h1 = self.args.n_hidden_nodes // 2
            h2 = self.args.n_hidden_nodes - h1

        for i in range(self.args.n_separate_layers):
            index = self.args.n_common_layers + i
            x1 = self.get_one_layer(h1, x1, index, 1)
            x2 = self.get_one_layer(h2, x2, index, 2)
        return x1, x2

    def get_structure(self):
        if len(self.input_shape) > 1:
            input_type = tf.float32
        else:
            input_type = tf.int32

        inputs = Input(shape=self.input_shape, dtype=input_type)
        x1, x2 = self.get_main_model(inputs)

        if self.args.loss_type == 'hinge':
            activation = 'linear'
        else:
            activation = 'softmax'
        outputs = [
            self.get_output_layer(x1, activation, 'y1'),
            self.get_output_layer(x2, activation, 'y2')
        ]
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def get_model(self):
        model = self.get_structure()
        adam = tf.keras.optimizers.Adam(lr=self.args.lr)
        if self.args.loss_type == 'hinge':
            loss = 'categorical_hinge'
        else:
            loss = 'categorical_crossentropy'
        model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
        return model

    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size


class AbstractSingleModelGenerator(AbstractModelGenerator):
    def get_main_model(self, x):
        x = self.convert_input(x)
        for i in range(self.args.n_common_layers):
            x = self.get_one_layer(self.args.n_hidden_nodes, x, i, 0)
        x_list = [x] * self.output_nodes

        if self.constant_n_hidden_nodes():
            h_size = self.args.n_hidden_nodes
        else:
            h_size = self.args.n_hidden_nodes // self.output_nodes
        h_list = [h_size] * self.output_nodes

        for i in range(self.args.n_separate_layers):
            index = self.args.n_common_layers + i
            x_list = [self.get_one_layer(h, x, index, j + 1) for j, [h, x] in
                      enumerate(zip(h_list, x_list))]
        return x_list

    def get_output_layer(self, x, activation, name):
        return Dense(1, activation=activation, name=name)(x)

    def get_structure(self):
        if len(self.input_shape) > 1:
            input_type = tf.float32
        else:
            input_type = tf.int32

        inputs = Input(shape=self.input_shape, dtype=input_type)
        x_list = self.get_main_model(inputs)

        activation = 'sigmoid'
        outputs = [self.get_output_layer(x, activation, 'y' + str(i + 1)) for
                   i, x in enumerate(x_list)]
        outputs = tf.concat(outputs, -1)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def get_model(self):
        model = self.get_structure()
        adam = tf.keras.optimizers.Adam(lr=self.args.lr)
        loss = tf.keras.losses.BinaryCrossentropy()
        model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
        return model
