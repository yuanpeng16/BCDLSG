import tensorflow as tf

from official.nlp.transformer import attention_layer
from official.nlp.transformer import embedding_layer
from official.nlp.transformer import ffn_layer
from official.nlp.transformer import model_utils

from transformer import Transformer
from transformer import EncoderStack
from transformer import PrePostProcessingWrapper
from abstract_model import AbstractModelGenerator


class SeparatedEncoderStack(EncoderStack):
    """Transformer encoder stack.

    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
      1. Self-attention layer
      2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self, params):
        super(EncoderStack, self).__init__()
        self.params = params
        self.common_layers = []
        self.separate_layers1 = []
        self.separate_layers2 = []

    def get_layers(self, n_layers):
        params = self.params
        layers = []
        for _ in range(n_layers):
            # Create sublayers for each layer.
            self_attention_layer = attention_layer.SelfAttention(
                params["hidden_size"], params["num_heads"],
                params["attention_dropout"])
            feed_forward_network = ffn_layer.FeedForwardNetwork(
                params["hidden_size"], params["filter_size"],
                params["relu_dropout"])

            layers.append([
                PrePostProcessingWrapper(self_attention_layer, params),
                PrePostProcessingWrapper(feed_forward_network, params)
            ])
        return layers

    def build(self, input_shape):
        """Builds the encoder stack."""
        params = self.params
        if params["n_shared_layers"] == 0:
            n_shared_layers = params["n_shared_layers"]
            n_individual_layers = params["n_individual_layers"] - 1
        else:
            n_shared_layers = params["n_shared_layers"] - 1
            n_individual_layers = params["n_individual_layers"]
        assert n_shared_layers + n_individual_layers > 0
        self.common_layers.extend(self.get_layers(n_shared_layers))
        self.separate_layers1.extend(self.get_layers(n_individual_layers))
        self.separate_layers2.extend(self.get_layers(n_individual_layers))

        # Create final layer normalization layer.
        self.output_list = []
        self.flat_list = []
        for _ in range(2):
            self.output_list.append(tf.keras.layers.LayerNormalization(
                epsilon=1e-6, dtype="float32"))
            self.flat_list.append(tf.keras.layers.Flatten())
        super(EncoderStack, self).build(input_shape)

    def call_layers(self, encoder_inputs, layers, attention_bias, training):
        for n, layer in enumerate(layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.name_scope("layer_%d" % n):
                with tf.name_scope("self_attention"):
                    encoder_inputs = self_attention_layer(
                        encoder_inputs, attention_bias, training=training)
                with tf.name_scope("ffn"):
                    encoder_inputs = feed_forward_network(
                        encoder_inputs, training=training)
        return encoder_inputs

    def call(self, encoder_inputs, attention_bias, inputs_padding, training):
        """Return the output of the encoder layer stacks.

        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
            1, input_length]
          inputs_padding: tensor with shape [batch_size, input_length], inputs with
            zero paddings.
          training: boolean, whether in training mode or not.

        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        if len(encoder_inputs) == 1:
            encoder_inputs = self.call_layers(encoder_inputs[0],
                                              self.common_layers,
                                              attention_bias, training)
            encoder_inputs1, encoder_inputs2 = encoder_inputs, encoder_inputs
        else:
            encoder_inputs1, encoder_inputs2 = encoder_inputs

        encoder_inputs1 = self.call_layers(encoder_inputs1,
                                           self.separate_layers1,
                                           attention_bias, training)
        encoder_inputs2 = self.call_layers(encoder_inputs2,
                                           self.separate_layers2,
                                           attention_bias, training)

        x1 = self.output_list[0](encoder_inputs1)
        x1 = self.flat_list[0](x1)
        x2 = self.output_list[1](encoder_inputs2)
        x2 = self.flat_list[1](x2)
        return x1, x2


class SeparateTransformer(Transformer):
    """Transformer model with Keras.

    Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

    The Transformer model consists of an encoder and decoder. The input is an int
    sequence (or a batch of sequences). The encoder produces a continuous
    representation, and the decoder uses the encoder output to generate
    probabilities for the output sequence.
    """

    def __init__(self, params, name=None):
        """Initialize layers to build Transformer model.

        Args:
          params: hyperparameter object defining layer sizes, dropout values, etc.
          name: name of the model.
        """
        super(Transformer, self).__init__(name=name)
        self.params = params
        self.softmax_list = []
        if params['n_shared_layers'] == 0:
            copies = 2
        else:
            copies = 1
        for _ in range(copies):
            self.softmax_list.append(embedding_layer.EmbeddingSharedWeights(
                params["vocab_size"], params["hidden_size"]))

        self.encoder_stack = SeparatedEncoderStack(params)

    def call(self, inputs, training):
        """Calculate target logits or inferred target sequences.

        Args:
          inputs: input tensor list of size 1 or 2.
            First item, inputs: int tensor with shape [batch_size, input_length].
            Second item (optional), targets: None or int tensor with shape
              [batch_size, target_length].
          training: boolean, whether in training mode or not.

        Returns:
          If targets is defined, then return logits for each word in the target
          sequence. float tensor with shape [batch_size, target_length, vocab_size]
          If target is none, then generate output sequence one token at a time.
            returns a dictionary {
              outputs: [batch_size, decoded length]
              scores: [batch_size, float]}
          Even when float16 is used, the output tensor(s) are always float32.

        Raises:
          NotImplementedError: If try to use padded decode method on CPU/GPUs.
        """
        if len(inputs) == 2:
            inputs, targets = inputs[0], inputs[1]
        else:
            # Decoding path.
            inputs, targets = inputs[0], None

        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        with tf.name_scope("Transformer"):
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias = model_utils.get_padding_bias(inputs)

            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            encoder_outputs = self.encode(inputs, attention_bias, training)
            return encoder_outputs

    def encode(self, inputs, attention_bias, training):
        """Generate continuous representation for inputs.

        Args:
          inputs: int tensor with shape [batch_size, input_length].
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length].
          training: boolean, whether in training mode or not.

        Returns:
          float tensor with shape [batch_size, input_length, hidden_size]
        """
        with tf.name_scope("encode"):
            # Prepare inputs to the layer stack by adding positional encodings and
            # applying dropout.
            # embedded_inputs = self.embedding_softmax_layer(inputs)
            embedded_inputs = [layer(inputs) for layer in self.softmax_list]
            embedded_inputs = [tf.cast(layer, self.params["dtype"]) for layer
                               in embedded_inputs]
            inputs_padding = model_utils.get_padding(inputs)
            attention_bias = tf.cast(attention_bias, self.params["dtype"])

            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(embedded_inputs[0])[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.params["hidden_size"])
                pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
                encoder_inputs = [layer + pos_encoding for layer in
                                  embedded_inputs]

            if training:
                encoder_inputs = [tf.nn.dropout(
                    layer,
                    rate=self.params["layer_postprocess_dropout"]) for layer in
                    embedded_inputs]

            return self.encoder_stack(
                encoder_inputs, attention_bias, inputs_padding,
                training=training)


def get_transformer_model(x, hn, n_shared_layers, n_individual_layers,
                          vocab_size):
    params = {
        'vocab_size': vocab_size,
        'hidden_size': hn,
        'dtype': tf.float32,
        'layer_postprocess_dropout': 0.5,
        'n_shared_layers': n_shared_layers,
        'n_individual_layers': n_individual_layers,
        'num_heads': 8,
        'attention_dropout': 0.5,
        'filter_size': hn,
        'relu_dropout': 0.5,
        'padded_decode': False
    }
    internal_model = SeparateTransformer(params, name="transformer_v2")
    return internal_model([x], training=False)


class TransformerGenerator(AbstractModelGenerator):
    def get_main_model(self, x):
        return get_transformer_model(
            x, self.args.n_hidden_nodes, self.args.n_shared_layers,
            self.args.n_individual_layers, self.vocab_size)
