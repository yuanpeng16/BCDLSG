import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from official.nlp.transformer import attention_layer

from abstract_model import AbstractModelGenerator

#https://keras.io/examples/vision/image_classification_with_vision_transformer

def gelu(features, approximate=False, name=None):
  """from tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/ops/nn_ops.py
  """
  with ops.name_scope(name, "Gelu", [features]):
    features = ops.convert_to_tensor(features, name="features")
    if approximate:
      coeff = math_ops.cast(0.044715, features.dtype)
      return 0.5 * features * (
          1.0 + math_ops.tanh(0.7978845608028654 *
                              (features + coeff * math_ops.pow(features, 3))))
    else:
      return 0.5 * features * (1.0 + math_ops.erf(
          features / math_ops.cast(1.4142135623730951, features.dtype)))


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def get_a_vision_transformer_layer(
        x, projection_dim, layer_id, layer_num, image_size):
    patch_size = 6  # Size of the patches to be extract from the input images
    if layer_id < 0:
        return Patches(patch_size)(x)
    if layer_id == 0:
        num_patches = (image_size // patch_size) ** 2
        return PatchEncoder(num_patches, projection_dim)(x)
    elif layer_id == layer_num - 1:
        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(x)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        # Size of the dense layers of the final classifier
        mlp_head_units = [2048, 1024]
        features = mlp(representation, hidden_units=mlp_head_units,
                       dropout_rate=0.5)
        return features
    else:
        num_heads = 4
        transformer_units = [
            projection_dim * 2,
            projection_dim,
        ]  # Size of the transformer layers
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = attention_layer.SelfAttention(
            projection_dim, num_heads, 0.1)(x1, 0)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        return layers.Add()([x3, x2])


class SeparateVisionTransformer(AbstractModelGenerator):
    def constant_n_hidden_nodes(self):
        return True

    def convert_input(self, x):
        return get_a_vision_transformer_layer(
            x, 0, -1, self.depth, self.input_shape[0])

    def get_one_layer(self, hn, x, index, part):
        return get_a_vision_transformer_layer(
            x, hn, index, self.depth, self.input_shape[0])
