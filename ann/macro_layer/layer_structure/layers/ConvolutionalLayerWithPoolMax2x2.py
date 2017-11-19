import tensorflow as tf
import numpy as np

from ann.macro_layer.layer_structure.layers.ConvolutionalLayer import ConvolutionalLayer


class ConvolutionalLayerWithPoolMax2x2(ConvolutionalLayer):
    def __init__(self, height_patch, width_patch, filters_amount, strides, padding='SAME', max_pool_padding='SAME'):
        super(ConvolutionalLayerWithPoolMax2x2, self).__init__(
            height_patch, width_patch, filters_amount, strides, padding)
        self.max_pool_padding = max_pool_padding
        self.pool_output = None

    def connect_layer(self, prev_layer, input_tensor):
        with tf.name_scope('PoolMax2x2'):
            super(ConvolutionalLayerWithPoolMax2x2, self).connect_layer(prev_layer, input_tensor)
            self.pool_output = tf.nn.max_pool(self.output, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding=self.max_pool_padding)

    def get_tensor(self):
        if self.pool_output is not None:
            return self.pool_output
        else:
            raise Exception("NotConnected")

    def calc_image_height_width(self, prev_layer):
        if self.padding == 'VALID':
            raise Exception('Case not implemented')  # ToDo: create a algorithm to calc h and w in padding=VALID case
        elif self.padding == 'SAME':
            self.height_image = np.int64(prev_layer.height_image / 2)
            self.width_image = np.int64(prev_layer.width_image / 2)
        else:
            raise Exception('padding name not supported')