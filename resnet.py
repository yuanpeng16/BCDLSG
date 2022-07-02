import tensorflow as tf

from official.vision.image_classification.resnet.resnet_model import \
    _gen_l2_regularizer
from official.vision.image_classification.resnet.resnet_model import \
    BATCH_NORM_DECAY
from official.vision.image_classification.resnet.resnet_model import \
    BATCH_NORM_EPSILON
from official.vision.image_classification.resnet.resnet_model import conv_block
from official.vision.image_classification.resnet.resnet_model import \
    identity_block
from official.vision.image_classification.resnet.resnet_model import \
    initializers
from abstract_model import AbstractModelGenerator

layers = tf.keras.layers


def stage1(x, prefix, scale, use_l2_regularizer=True):
    bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3, 3), name=prefix + 'conv1_pad')(x)
    x = layers.Conv2D(
        64 // scale, (7, 7),
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=prefix + 'conv1')(
        x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=prefix + 'bn_conv1')(
        x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    return x


def stage2(x, prefix, scale, use_l2_regularizer=True):
    x = conv_block(
        x,
        3, [64 // scale, 64 // scale, 256 // scale],
        stage=2,
        block=prefix + 'a',
        strides=(1, 1),
        use_l2_regularizer=use_l2_regularizer)
    x = identity_block(
        x,
        3, [64 // scale, 64 // scale, 256 // scale],
        stage=2,
        block=prefix + 'b',
        use_l2_regularizer=use_l2_regularizer)
    x = identity_block(
        x,
        3, [64 // scale, 64 // scale, 256 // scale],
        stage=2,
        block=prefix + 'c',
        use_l2_regularizer=use_l2_regularizer)
    return x


def stage3(x, prefix, scale, use_l2_regularizer=True):
    x = conv_block(
        x,
        3, [128 // scale, 128 // scale, 512 // scale],
        stage=3,
        block=prefix + 'a',
        use_l2_regularizer=use_l2_regularizer)
    x = identity_block(
        x,
        3, [128 // scale, 128 // scale, 512 // scale],
        stage=3,
        block=prefix + 'b',
        use_l2_regularizer=use_l2_regularizer)
    x = identity_block(
        x,
        3, [128 // scale, 128 // scale, 512 // scale],
        stage=3,
        block=prefix + 'c',
        use_l2_regularizer=use_l2_regularizer)
    x = identity_block(
        x,
        3, [128 // scale, 128 // scale, 512 // scale],
        stage=3,
        block=prefix + 'd',
        use_l2_regularizer=use_l2_regularizer)
    return x


def stage4(x, prefix, scale, use_l2_regularizer=True):
    x = conv_block(
        x,
        3, [256 // scale, 256 // scale, 1024 // scale],
        stage=4,
        block=prefix + 'a',
        use_l2_regularizer=use_l2_regularizer)
    x = identity_block(
        x,
        3, [256 // scale, 256 // scale, 1024 // scale],
        stage=4,
        block=prefix + 'b',
        use_l2_regularizer=use_l2_regularizer)
    x = identity_block(
        x,
        3, [256 // scale, 256 // scale, 1024 // scale],
        stage=4,
        block=prefix + 'c',
        use_l2_regularizer=use_l2_regularizer)
    x = identity_block(
        x,
        3, [256 // scale, 256 // scale, 1024 // scale],
        stage=4,
        block=prefix + 'd',
        use_l2_regularizer=use_l2_regularizer)
    x = identity_block(
        x,
        3, [256 // scale, 256 // scale, 1024 // scale],
        stage=4,
        block=prefix + 'e',
        use_l2_regularizer=use_l2_regularizer)
    x = identity_block(
        x,
        3, [256 // scale, 256 // scale, 1024 // scale],
        stage=4,
        block=prefix + 'f',
        use_l2_regularizer=use_l2_regularizer)
    return x


def stage5(x, prefix, scale, use_l2_regularizer=True):
    x = conv_block(
        x,
        3, [512 // scale, 512 // scale, 2048 // scale],
        stage=5,
        block=prefix + 'a',
        use_l2_regularizer=use_l2_regularizer)
    x = identity_block(
        x,
        3, [512 // scale, 512 // scale, 2048 // scale],
        stage=5,
        block=prefix + 'b',
        use_l2_regularizer=use_l2_regularizer)
    x = identity_block(
        x,
        3, [512 // scale, 512 // scale, 2048 // scale],
        stage=5,
        block=prefix + 'c',
        use_l2_regularizer=use_l2_regularizer)

    x = layers.GlobalAveragePooling2D()(x)
    return x


def get_stage(x, prefix, scale, stage, use_l2_regularizer=True):
    assert stage >= 0
    assert stage < 5
    if stage == 0:
        return stage1(x, prefix, scale, use_l2_regularizer)
    elif stage == 1:
        x = stage2(x, prefix, scale, use_l2_regularizer)
    elif stage == 2:
        x = stage3(x, prefix, scale, use_l2_regularizer)
    elif stage == 3:
        x = stage4(x, prefix, scale, use_l2_regularizer)
    elif stage == 4:
        x = stage5(x, prefix, scale, use_l2_regularizer)
    return x


def get_output_layer(x, prefix, num_classes=10,
                     ff_activation='softmax', use_l2_regularizer=True):
    x = layers.Dense(
        num_classes,
        activation=ff_activation,
        kernel_initializer=initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=prefix + 'fc1000')(
        x)
    return x


class ResNetGenerator(AbstractModelGenerator):
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
