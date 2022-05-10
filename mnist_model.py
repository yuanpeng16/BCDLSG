import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
import numpy as np


def get_model_generator(args, input_shape, output_nodes):
    if args.model_type == 'dnn':
        model = DeepModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'cnn':
        model = CNNModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'residual':
        model = ResidualModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'residual_cnn':
        model = ResidualCNNModelGenerator(args, input_shape, output_nodes)
    else:
        assert False
    return model


class DeepModelGenerator(object):
    def __init__(self, args, input_shape, output_nodes):
        self.args = args
        self.input_shape = input_shape
        self.output_nodes = output_nodes

    def get_one_layer(self, hn, x):
        return tf.keras.layers.Dense(hn, activation='relu')(x)

    def get_core_structure(self, hn, n_common_layers, n_separate_layers, x):
        for _ in range(n_common_layers):
            x = self.get_one_layer(hn, x)
        x1, x2 = x, x
        h1 = int(hn / 2)
        h2 = hn - h1
        for _ in range(n_separate_layers):
            x1 = self.get_one_layer(h1, x1)
            x2 = self.get_one_layer(h2, x2)
        return x1, x2

    def get_main_model(self, x):
        x = tf.keras.layers.Flatten()(x)
        x1, x2 = self.get_core_structure(self.args.n_hidden_nodes,
                                         self.args.n_common_layers,
                                         self.args.n_separate_layers, x)
        return x1, x2

    def get_structure(self):
        inputs = Input(shape=self.input_shape)
        x1, x2 = self.get_main_model(inputs)

        if self.args.loss_type == 'hinge':
            activation = 'linear'
        else:
            activation = 'softmax'
        outputs = [
            Dense(self.output_nodes, activation=activation, name='y1')(x1),
            Dense(self.output_nodes, activation=activation, name='y2')(x2)
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


class CNNModelGenerator(DeepModelGenerator):
    def get_one_layer(self, hn, x):
        return tf.keras.layers.Conv2D(hn, (3, 3), activation='relu',
                                      padding='SAME')(x)

    def post_layers(self, hn, x):
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(hn, activation='relu')(x)
        return x

    def get_main_model(self, x):
        hn = self.args.n_hidden_nodes
        if self.args.n_separate_layers == 0:
            n_common_layers = self.args.n_common_layers - 1
            n_separate_layers = self.args.n_separate_layers
        else:
            n_common_layers = self.args.n_common_layers
            n_separate_layers = self.args.n_separate_layers - 1

        x1, x2 = self.get_core_structure(self.args.n_hidden_nodes,
                                         n_common_layers,
                                         n_separate_layers, x)
        x1 = self.post_layers(hn, x1)
        if self.args.n_separate_layers == 0:
            x2 = x1
        else:
            x2 = self.post_layers(hn, x2)
        return x1, x2


def residual(x, layer):
    comparison = np.array(x.shape) == np.array(layer.shape)
    if comparison.all():
        layer += x
    return layer


class ResidualModelGenerator(DeepModelGenerator):
    def get_one_layer(self, hn, x):
        layer = super().get_one_layer(hn, x)
        return residual(x, layer)


class ResidualCNNModelGenerator(CNNModelGenerator):
    def get_one_layer(self, hn, x):
        layer = super().get_one_layer(hn, x)
        return residual(x, layer)
