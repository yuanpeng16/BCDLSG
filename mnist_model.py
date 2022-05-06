import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense


def get_model_generator(args, input_shape, output_nodes):
    return DeepModelGenerator(args, input_shape, output_nodes)


class DeepModelGenerator(object):
    def __init__(self, args, input_shape, output_nodes):
        self.args = args
        self.input_shape = input_shape
        self.output_nodes = output_nodes

    def get_structure(self):
        inputs = Input(shape=self.input_shape)
        x = inputs
        for _ in range(1):
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        for _ in range(2):
            x = tf.keras.layers.Dense(64, activation='relu')(x)

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
