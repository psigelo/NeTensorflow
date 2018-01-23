import json
import numpy as np
import tensorflow as tf
import uuid
import os

from netensorflow.ann.ANNGlobals import register_netensorflow_class
from netensorflow.ann.macro_layer.layer_structure.layers.ConvolutionalLayer import ConvolutionalLayer


@register_netensorflow_class
class ConvolutionalLayerWithPoolNorm(ConvolutionalLayer):
    def __init__(self, height_patch=None, width_patch=None, filters_amount=None, strides=list([1, 1, 1, 1]),
                 padding='SAME', max_pool_padding='SAME', restore=False):
        self.name = self.__class__.__name__ + '_uuid_' + uuid.uuid4().hex
        self.__max_pool_padding = None
        self.__norm_output = None
        if not restore:
            super(ConvolutionalLayerWithPoolNorm, self).__init__(
                height_patch, width_patch, filters_amount, strides, padding)
            self.max_pool_padding = max_pool_padding
        else:
            super(ConvolutionalLayerWithPoolNorm, self).__init__(restore=True)

    def connect_layer(self, prev_layer, input_tensor):
        with tf.name_scope('PoolMax2x2'):
            super(ConvolutionalLayerWithPoolNorm, self).connect_layer(prev_layer, input_tensor)
            pool_output = tf.nn.max_pool(self.output, ksize=[1, 2, 2, 1],
                                              strides=[1, 2, 2, 1], padding=self.max_pool_padding)
            self.norm_output = tf.nn.lrn(pool_output, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    def get_tensor(self):
        if self.norm_output is not None:
            return self.norm_output
        else:
            raise Exception("NotConnected")

    def save_netensorflow_model(self, path):
        layer_path = os.path.join(path, self.name)
        with open(layer_path + '_internal_data.json', 'w') as fp:
            json.dump(self.save_and_restore_dictionary, fp)

    @property
    def max_pool_padding(self):
        return self.__max_pool_padding

    @max_pool_padding.setter
    def max_pool_padding(self, max_pool_padding):
        self.__max_pool_padding = max_pool_padding
        self.save_and_restore_dictionary['max_pool_padding'] = self.__max_pool_padding

    @property
    def norm_output(self):
        return self.__norm_output

    @norm_output.setter
    def norm_output(self, norm_output):
        if isinstance(norm_output, str):
            norm_output = tf.get_default_graph().get_tensor_by_name(norm_output)
        self.__norm_output = norm_output
        self.save_and_restore_dictionary['norm_output'] = self.__norm_output.name

    @classmethod
    def restore_netensorflow_model(cls, path, name):
        layer_path = os.path.join(path, name)
        with open(layer_path + '_internal_data.json', 'r') as fp:
            restore_json_dict = json.load(fp)

        layer = cls(restore=True)
        for var_name in restore_json_dict:
            setattr(layer, var_name, restore_json_dict[var_name])
        layer.name = name
        return layer
