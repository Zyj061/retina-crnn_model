from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from tensorflow.keras.layers import AveragePooling1D
import tensorflow as tf

class ParametricSoftplus(Layer):
    '''
    Parametric Softplus of the form: alpha * log(1 + exp(beta * x))

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        alpha_init: float. Initial value of the alpha weights.
        beta_init: float. Initial values of the beta weights.
        weights: initial weights, as a list of 2 numpy arrays.

    # References:
        - [Inferring Nonlinear Neuronal Computation Based on Physiologically Plausible Inputs](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003143)
    '''
    def __init__(self, alpha_init=0.2, beta_init=5.0,
                 weights=None, **kwargs):

        super(ParametricSoftplus, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.initial_weights = weights

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        input_shape = input_shape[1:]
        self.param_broadcast = [False] * len(param_shape)
        '''
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i-1] = 1
                self.param_broadcast[i-1] = True
        '''

        self.alphas = K.variable(self.alpha_init * np.ones(input_shape))
        self.betas = K.variable(self.beta_init * np.ones(input_shape))
        self.trainable_weights = [self.alphas, self.betas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.build = True

    def call(self, inputs, mask=None):
        X = inputs
        return K.softplus(self.betas * X) * self.alphas

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha_init": self.alpha_init,
                  "beta_init": self.beta_init}
        base_config = super(ParametricSoftplus, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class lateral_layer(Layer):

    def __init__(self, group_size, **kwargs):
        super(lateral_layer, self).__init__(**kwargs)
        self.group_size = group_size

    def build(self, input_shape):
        self.neuron_num = int(input_shape[1])

        self.index = np.arange(self.neuron_num, dtype=np.int32)
        np.random.shuffle(self.index)

        assert self.neuron_num%self.group_size==0, "group size must be divided by the number of RGC"
        self.group_num = int((self.neuron_num / self.group_size))
        print('group number: ' + str(self.group_num))

        self.build = True

    def call(self, inputs):

        self.normalized_output = tf.Variable(tf.zeros_like(inputs), dtype='float32', name='lateral_output', validate_shape=True)

        for i_group in range(self.group_num):
            begin_index = int(i_group * self.group_size)
            end_index = int(begin_index + self.group_size)
            tmp_index = self.index[begin_index:end_index]

            tmp_x = tf.gather(inputs, tmp_index, axis=-1)

            norm_value = K.sum(tmp_x, axis=1, keepdims=True)
            norm_value = K.stack([norm_value] * self.group_size, axis=1)
            norm_value = tf.squeeze(norm_value, axis=-1)
            norm_x = tf.divide(tmp_x, norm_value)

            col_indices_nd1, col_indices_nd2 = tf.meshgrid(tf.range(tf.shape(self.normalized_output)[0]), tmp_index, indexing='ij')

            row_tf = tf.reshape(col_indices_nd1, shape=[-1, 1])
            col_tf = tf.reshape(col_indices_nd2, shape=[-1, 1])
            sparse_indicies = tf.reshape(tf.concat([row_tf, col_tf], axis=-1), shape=[-1,2])
            norm_x = tf.reshape(norm_x, [-1])

            self.normalized_output = tf.scatter_nd_update(self.normalized_output, tf.cast(sparse_indicies, tf.int32), norm_x)

        results = tf.convert_to_tensor(self.normalized_output)
        results.set_shape((None, self.neuron_num))
        return results

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "group_size": self.group_size}
        base_config = super(lateral_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.neuron_num)


