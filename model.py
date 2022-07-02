import tensorflow as tf

from abstract_model import AbstractModelGenerator
from resnet import get_stage, get_output_layer
from separate_transformer import get_transformer_model
from separate_vision_transformer import get_a_vision_transformer_layer


def get_model_generator(args, input_shape, output_nodes):
    if args.model_type == 'dnn':
        model = DNNModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'cnn':
        model = CNNModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'resnet':
        model = SeparatedResNet(args, input_shape, output_nodes)
    elif args.model_type == 'vision_transformer':
        model = SeparateVisionTransformer(args, input_shape, output_nodes)
    elif args.model_type == 'transformer':
        model = SeparateTransformer(args, input_shape, output_nodes)
    elif args.model_type == 'lstm':
        model = LSTMModelGenerator(args, input_shape, output_nodes)
    else:
        assert False
    return model


class DNNModelGenerator(AbstractModelGenerator):
    def convert_input(self, x):
        return tf.keras.layers.Flatten()(x)

    def get_intermediate_layer(self, hn, x):
        return tf.keras.layers.Dense(hn, activation='relu')(x)


class CNNModelGenerator(AbstractModelGenerator):
    def get_one_layer(self, hn, x, index, part):
        if index == self.depth - 1:
            x = tf.keras.layers.Dense(2 * hn, activation='relu')(x)
        else:
            x = tf.keras.layers.Conv2D(hn, (3, 3), activation='relu',
                                       padding='SAME')(x)
            if index == self.depth - 2:
                x = tf.keras.layers.Flatten()(x)
        return x


class LSTMModelGenerator(AbstractModelGenerator):
    def get_intermediate_layer(self, hn, x):
        hidden_size = hn // 2
        return tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, return_sequences=True))(x)

    def get_first_layer(self, hn, x):
        return tf.keras.layers.Embedding(self.vocab_size, hn)(x)

    def get_last_layer(self, hn, x):
        x = self.get_intermediate_layer(hn, x)
        return tf.keras.layers.Flatten()(x)


class SeparatedResNet(AbstractModelGenerator):
    def __init__(self, args, input_shape, output_nodes):
        super().__init__(args, input_shape, output_nodes)
        assert self.depth == 5

    def get_output_layer(self, x, activation, name):
        return get_output_layer(x, name, num_classes=self.output_nodes,
                                ff_activation=activation,
                                use_l2_regularizer=False)

    def get_one_layer(self, hn, x, index, part):
        if part == 0:
            scale = 1
        else:
            scale = 2
        prefix = 'Net' + str(part)
        return get_stage(x, prefix, scale, index, use_l2_regularizer=False)


class SeparateVisionTransformer(AbstractModelGenerator):
    def constant_n_hidden_nodes(self):
        return True

    def convert_input(self, x):
        return get_a_vision_transformer_layer(
            x, 0, -1, self.depth, self.input_shape[0])

    def get_one_layer(self, hn, x, index, part):
        return get_a_vision_transformer_layer(
            x, hn, index, self.depth, self.input_shape[0])


class SeparateTransformer(AbstractModelGenerator):
    def get_main_model(self, x):
        return get_transformer_model(
            x, self.args.n_hidden_nodes, self.args.n_common_layers,
            self.args.n_separate_layers, self.vocab_size)
