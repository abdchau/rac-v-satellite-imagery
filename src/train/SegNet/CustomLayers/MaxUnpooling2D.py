'''
Created on Jan 30, 2019

@author: daniel
'''
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow import scatter_nd

class MaxUnpooling2D(Layer):
    def __init__(self, pool_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'pool_size': self.pool_size
        })
        return config

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        # with K.variable_scope(self.name):
        mask = K.cast(mask, 'int32')
        input_shape = K.shape(updates)

        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.pool_size[0],
                input_shape[2] * self.pool_size[1],
                input_shape[3])

        ret = scatter_nd(K.expand_dims(K.flatten(mask)),
                              K.flatten(updates),
                              [K.prod(output_shape)])

        input_shape = updates.shape
        out_shape = [-1,
                     input_shape[1] * self.pool_size[0],
                     input_shape[2] * self.pool_size[1],
                     input_shape[3]]
        return K.reshape(ret, out_shape)

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
                mask_shape[0],
                mask_shape[1]*self.pool_size[0],
                mask_shape[2]*self.pool_size[1],
                mask_shape[3]
                )