import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense


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

    def get_main_model(self, x):
        hn = self.args.n_hidden_nodes
        x = tf.keras.layers.Flatten()(x)
        for _ in range(self.args.n_hidden_layers):
            x = tf.keras.layers.Dense(hn, activation='relu')(x)
        return x

    def get_structure(self):
        inputs = Input(shape=self.input_shape)
        x = self.get_main_model(inputs)

        if self.args.loss_type == 'hinge':
            activation = 'linear'
        else:
            activation = 'softmax'
        outputs = [
            Dense(self.output_nodes, activation=activation, name='y1')(x),
            Dense(self.output_nodes, activation=activation, name='y2')(x)
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
    def get_main_model(self, x):
        hn = self.args.n_hidden_nodes
        x = tf.keras.layers.Conv2D(hn, (3, 3), activation='relu')(x)
        for _ in range(self.args.n_hidden_layers):
            x = tf.keras.layers.Conv2D(hn, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(2 * hn, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        for _ in range(1):
            x = tf.keras.layers.Dense(hn, activation='relu')(x)
        return x


class ResidualModelGenerator(DeepModelGenerator):
    def get_main_model(self, x):
        hn = self.args.n_hidden_nodes
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(hn, activation='relu')(x)
        for _ in range(self.args.n_hidden_layers):
            x += tf.keras.layers.Dense(hn, activation='relu')(x)
        return x


class ResidualCNNModelGenerator(DeepModelGenerator):
    def get_main_model(self, x):
        hn = self.args.n_hidden_nodes
        x = tf.keras.layers.Conv2D(hn, (3, 3), activation='relu')(x)
        for _ in range(self.args.n_hidden_layers):
            x += tf.keras.layers.Conv2D(hn, (3, 3), activation='relu', padding='SAME')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(2 * hn, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        for _ in range(1):
            x = tf.keras.layers.Dense(hn, activation='relu')(x)
        return x
