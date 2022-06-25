import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
import numpy as np

from resnet import get_resnet_model
from separate_transformer import get_transformer_model
from separate_vision_transformer import get_a_vision_transformer_layer


def get_model_generator(args, input_shape, output_nodes):
    if args.model_type == 'dnn':
        model = DeepModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'cnn':
        model = CNNModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'residual':
        model = ResidualModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'residual_cnn':
        model = ResidualCNNModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'resnet':
        model = SeparatedResNet(args, input_shape, output_nodes)
    elif args.model_type == 'vision_transformer':
        model = SeparateVisionTransformer(args, input_shape, output_nodes)
    elif args.model_type == 'transformer':
        model = SeparateTransformer(args, input_shape, output_nodes)
    elif args.model_type == 'lstm':
        model = LSTMModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'gnn':
        model = GNNModelGenerator(args, input_shape, output_nodes)
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

    def get_input_type(self):
        return tf.float32

    def get_structure(self):
        inputs = Input(shape=self.input_shape, dtype=self.get_input_type())
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

    def set_vocab_size(self, vocab_size):
        # Do nothing
        pass


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
        if self.args.n_separate_layers == 0:
            x1 = self.post_layers(2 * hn, x1)
            x2 = x1
        else:
            x1 = self.post_layers(hn, x1)
            x2 = self.post_layers(hn, x2)
        return x1, x2


class GNNModelGenerator(CNNModelGenerator):
    def get_one_layer(self, hn, x):
        h = tf.keras.layers.Conv2D(hn, (3, 1), (1, 1), activation=None,
                                   padding='SAME')(x)
        v = tf.keras.layers.Conv2D(hn, (1, 3), (1, 1), activation=None,
                                   padding='SAME')(x)
        x = h + v
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def get_one_layer1(self, hn, x):
        x = tf.keras.layers.AveragePooling2D((3, 3), (1, 1), padding='SAME')(x)
        x = tf.keras.layers.Conv2D(hn, (1, 1), activation='relu',
                                   padding='SAME')(x)
        return x


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


class SeparatedResNet(DeepModelGenerator):
    def get_core_structure(self, x, stage, num_classes, ff_activation):
        x = get_resnet_model(x, 'A', 1, True, stage, use_l2_regularizer=False)
        x1 = get_resnet_model(x, 'B', 2, False, stage, num_classes,
                              ff_activation, use_l2_regularizer=False)
        x2 = get_resnet_model(x, 'C', 2, False, stage, num_classes,
                              ff_activation, use_l2_regularizer=False)
        return x1, x2

    def get_structure(self):
        assert self.args.n_common_layers + self.args.n_separate_layers == 5
        inputs = Input(shape=self.input_shape)
        if self.args.loss_type == 'hinge':
            ff_activation = 'linear'
        else:
            ff_activation = 'softmax'
        x1, x2 = self.get_core_structure(inputs, self.args.n_common_layers,
                                         self.output_nodes, ff_activation)
        outputs = [x1, x2]
        model = Model(inputs=inputs, outputs=outputs)
        return model


class SeparateTransformer(DeepModelGenerator):
    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size

    def get_input_type(self):
        return tf.int32

    def get_main_model(self, x):
        return get_transformer_model(
            x, self.args.n_hidden_nodes, self.args.n_common_layers,
            self.args.n_separate_layers, self.vocab_size)


class LSTMModelGenerator(DeepModelGenerator):
    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size

    def get_input_type(self):
        return tf.int32

    def get_one_layer(self, hn, x, layer_id):
        if layer_id == 0:
            x = tf.keras.layers.Embedding(self.vocab_size, hn)(x)
        else:
            hidden_size = int(hn / 2)
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(hidden_size, return_sequences=True))(x)
        if layer_id == self.args.n_common_layers + self.args.n_separate_layers - 1:
            x = tf.keras.layers.Flatten()(x)
        return x

    def get_core_structure(self, hn, n_common_layers, n_separate_layers, x):
        for i in range(n_common_layers):
            x = self.get_one_layer(hn, x, i)
        x1, x2 = x, x
        h1 = int(hn / 2)
        h2 = hn - h1
        for i in range(n_separate_layers):
            layer_id = n_common_layers + i
            x1 = self.get_one_layer(h1, x1, layer_id)
            x2 = self.get_one_layer(h2, x2, layer_id)
        return x1, x2

    def get_main_model(self, x):
        x1, x2 = self.get_core_structure(self.args.n_hidden_nodes,
                                         self.args.n_common_layers,
                                         self.args.n_separate_layers, x)
        return x1, x2


class SeparateVisionTransformer(DeepModelGenerator):
    def get_one_layer(self, hn, x, layer_id, num_layers):
        return get_a_vision_transformer_layer(
            x, hn, layer_id, num_layers, self.input_shape[0])

    def get_core_structure(self, hn, n_common_layers, n_separate_layers, x):
        num_layers = n_common_layers + n_separate_layers
        x = self.get_one_layer(hn, x, -1, num_layers)
        for i in range(n_common_layers):
            x = self.get_one_layer(hn, x, i, num_layers)
        x1, x2 = x, x
        h1 = hn
        h2 = hn
        for i in range(n_separate_layers):
            layer_id = n_common_layers + i
            x1 = self.get_one_layer(h1, x1, layer_id, num_layers)
            x2 = self.get_one_layer(h2, x2, layer_id, num_layers)
        return x1, x2

    def get_main_model(self, x):
        x1, x2 = self.get_core_structure(self.args.n_hidden_nodes,
                                         self.args.n_common_layers,
                                         self.args.n_separate_layers, x)
        return x1, x2