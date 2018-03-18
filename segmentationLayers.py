from keras import backend as K
from keras.layers import Layer
import tensorflow as tf

class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, pool_indices = tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        pool_indices = K.cast(pool_indices, K.floatx())
        return [output, pool_indices]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        out1 = output_shape[0], output_shape[1], output_shape[2], output_shape[3]
        out2 = output_shape[0], output_shape[1], output_shape[2], output_shape[3]
        return [out1, out2]





class UpSamplingWithArgmax2D(Layer):
    def __init__(self, **kwargs):
        super(UpSamplingWithArgmax2D, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """inputs = [input, pooling_indices]"""
        input, indices = inputs[0], inputs[1]
        if K.backend() == 'tensorflow':
            input_shape = tf.to_int32(tf.shape(input))
            output_shape = tf.to_int32([input_shape[0], input_shape[1]*2, input_shape[2]*2,input_shape[3]])
            indices = tf.to_int32(indices)
            x_indices = indices // (output_shape[2] * output_shape[3])
            x_indices = tf.to_int32(tf.reshape(x_indices, (-1, 1)))
            y_indices = indices % (output_shape[2] * output_shape[3]) // output_shape[3]
            y_indices = tf.to_int32(tf.reshape(y_indices, (-1, 1)))
            t1 = tf.range(input_shape[0])
            t1 = tf.reshape(t1, [-1, 1])
            t1 = tf.tile(t1, [1, input_shape[1] * input_shape[2] * input_shape[3]])
            t1 = tf.reshape(t1, [-1, 1])
            t2 = tf.range(input_shape[3])
            t2 = tf.reshape(t2, [-1, 1])
            t2 = tf.tile(t2, [input_shape[1] * input_shape[2] * input_shape[0], 1])
            unraveled_indices = tf.to_int32(tf.concat([t1, x_indices, y_indices, t2], -1))
            delta = tf.SparseTensor(tf.to_int64(unraveled_indices), tf.reshape(input, [-1]), tf.to_int64(output_shape))
            output = tf.sparse_add(tf.zeros(output_shape), delta)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        output = K.cast(output, K.floatx())
        return output

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim * ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape[0])]
        return (output_shape[0], output_shape[1], output_shape[2], output_shape[3])