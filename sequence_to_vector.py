# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
# from tensorflow.keras import layers, models
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)

        # TODO(students): start
        self.num_layers=num_layers
        self.layer=layers.Dense(input_dim,activation='relu')
        self.dropout=dropout
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        if(training):
            uniform_matrix=tf.random.uniform(tf.shape(sequence_mask),minval=0.0,maxval=1,dtype=tf.dtypes.float32)
            uniform_matrix=uniform_matrix > self.dropout
            uniform_matrix=tf.cast(uniform_matrix,tf.float32)
            sequence_mask=tf.multiply(sequence_mask,uniform_matrix)

        sequence_mask_reshaped = tf.reshape(sequence_mask, [sequence_mask.shape[0], sequence_mask.shape[1], 1])
        vector_sequence=tf.multiply(vector_sequence,sequence_mask_reshaped)
        sum = tf.reduce_sum(vector_sequence, 1, keepdims=True)
        sum = tf.reshape(sum, [sum.shape[0], sum.shape[2]])
        mask = tf.reduce_sum(sequence_mask, 1, keepdims=True)
        mask = tf.reshape(mask, mask.get_shape().as_list())
        avg = tf.math.divide_no_nan(sum, mask)

        layer_representations = []
        input = avg
        for j in range(self.num_layers):
            current_layer = self.layer(input)
            layer_representations.append(current_layer)
            input = current_layer

        combined_vector = current_layer

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}




class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)

        # TODO(students): start
        self.num_layers=num_layers
        self.layer=layers.GRU(self._input_dim,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        mask = tf.reshape(sequence_mask, [sequence_mask.shape[0],sequence_mask.shape[1],1])
        layer_representations = []
        input = vector_sequence
        for i in range(self.num_layers):
            current_sequence,layer_state = self.layer(input,mask=mask)
            layer_representations.append(layer_state)
            input = current_sequence

        combined_vector = layer_state
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
