import json

import os
import tensorflow as tf
import uuid

from netensorflow.ann.ANNGlobals import register_netensorflow_class
from netensorflow.ann.macro_layer.layer_structure.LayerStructure import LayerType


@register_netensorflow_class
class TranslatorLayerImage2OneDimension(object):
    def __init__(self):
        self.name = self.__class__.__name__ + '_uuid_' + uuid.uuid4().hex
        self.save_and_restore_dictionary = dict()
        self.__inputs_amount = None
        self.__output = None
        self.__layer_structure_name = None
        self.__summaries = list()

    def get_tensor(self):
        if self.output is not None:
            return self.output
        else:
            raise (ValueError, "FullConnected Layer not connected, output does not exists")

    def connect_layer(self, prev_layer, input_tensor):
        if prev_layer.layer_type != LayerType.IMAGE:
            raise Exception('PrevLayerMustBeTypeImage')

        self.inputs_amount = prev_layer.height_image * prev_layer.width_image * prev_layer.filters_amount
        with tf.name_scope('TranslatorLayerImage2OneDimension'):
            self.output = tf.reshape(input_tensor, [-1, self.inputs_amount])

    def save_netensorflow_model(self, path):
        layer_path = os.path.join(path, self.name)
        with open(layer_path + '_data.json', 'w') as fp:
            json.dump(self.save_and_restore_dictionary, fp)

    @property
    def layer_variables(self):
        return list()

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, output):
        self.__output = output
        self.save_and_restore_dictionary['output'] = self.__output.name

    @property
    def inputs_amount(self):
        return self.__inputs_amount

    @inputs_amount.setter
    def inputs_amount(self, inputs_amount):
        self.__inputs_amount = inputs_amount
        self.save_and_restore_dictionary['inputs_amount'] = self.__inputs_amount

    @property
    def summaries(self):
        return self.__summaries

    @summaries.setter
    def summaries(self, summaries):
        self.__summaries = summaries
        self.save_and_restore_dictionary['summaries'] = [summary.name for summary in self.__summaries]

    @property
    def layer_structure_name(self):
        return self.__layer_structure_name

    @layer_structure_name.setter
    def layer_structure_name(self, layer_structure_name):
        self.__layer_structure_name = layer_structure_name
        self.save_and_restore_dictionary['layer_structure_name'] = self.__layer_structure_name
