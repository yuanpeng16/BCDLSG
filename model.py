import tensorflow as tf

from abstract_model import AbstractModelGenerator
from abstract_model import AbstractSingleModelGenerator
from resnet import ResNetGenerator
from separate_transformer import TransformerGenerator
from separate_vision_transformer import VisionTransformerGenerator


def get_model_generator(args, input_shape, output_nodes):
    if args.model_type == 'dnn':
        model = DNNModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'dnn_dropout':
        model = DropoutDNNModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'cnn':
        model = CNNModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'resnet':
        model = ResNetGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'vision_transformer':
        model = VisionTransformerGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'lstm':
        model = LSTMModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'transformer':
        model = TransformerGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'dnn_single':
        model = DNNSingleModelGenerator(args, input_shape, output_nodes)
    elif args.model_type == 'cnn_single':
        model = CNNSingleModelGenerator(args, input_shape, output_nodes)
    else:
        raise ValueError(
            '{0} is not a valid model_type.'.format(args.model_type))
    return model


class DNNModelGenerator(AbstractModelGenerator):
    def convert_input(self, x):
        return tf.keras.layers.Flatten()(x)

    def get_intermediate_layer(self, hn, x):
        return tf.keras.layers.Dense(hn, activation='relu')(x)


class DropoutDNNModelGenerator(DNNModelGenerator):
    def get_intermediate_layer(self, hn, x):
        x = super().get_intermediate_layer(hn, x)
        return tf.keras.layers.Dropout(rate=0.1)(x)


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


class DNNSingleModelGenerator(AbstractSingleModelGenerator):
    def convert_input(self, x):
        return tf.keras.layers.Flatten()(x)

    def get_intermediate_layer(self, hn, x):
        return tf.keras.layers.Dense(hn, use_bias=False, activation='relu')(x)


class CNNSingleModelGenerator(AbstractSingleModelGenerator):
    def get_one_layer(self, hn, x, index, part):
        if index == self.depth - 1:
            x = tf.keras.layers.Dense(
                2 * hn, use_bias=False, activation='relu')(x)
        else:
            x = tf.keras.layers.Conv2D(hn, (3, 3), use_bias=False,
                                       activation='relu', padding='SAME')(x)
            if index == self.depth - 2:
                x = tf.keras.layers.Flatten()(x)
        return x
