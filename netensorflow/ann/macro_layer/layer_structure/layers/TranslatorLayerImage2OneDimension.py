import json

import os
from functools import reduce

import tensorflow as tf
import uuid

from netensorflow.ann.ANNGlobals import register_netensorflow_class
from netensorflow.ann.macro_layer.layer_structure.LayerStructure import LayerType
from netensorflow.ann.macro_layer.layer_structure.layers.AbsctractLayer import AbstractLayer


@register_netensorflow_class
class TranslatorLayerImage2OneDimension(AbstractLayer):
    def __init__(self, restore=False):
        super(TranslatorLayerImage2OneDimension, self).__init__()
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

        self.inputs_amount = int(reduce((lambda x,y: x*y) ,input_tensor.shape[1:]))
        with tf.name_scope('TranslatorLayerImage2OneDimension'):
            self.output = tf.reshape(input_tensor, [-1, self.inputs_amount])

    def save_netensorflow_model(self, path):
        layer_path = os.path.join(path, self.name)
        with open(layer_path + '_internal_data.json', 'w') as fp:
            json.dump(self.save_and_restore_dictionary, fp)

    @property
    def layer_variables(self):
        return list()

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, output):
        if isinstance(output, str):
            output = tf.get_default_graph().get_tensor_by_name(output)
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
        summaries_ = None
        if len(summaries) > 0:
            if isinstance(summaries[0], str):  # then is restoring, and is in string format.
                summaries_ = [tf.get_default_graph().get_tensor_by_name(summary) for summary in summaries]
        if summaries_ is None:
            summaries_ = summaries
        self.__summaries = summaries_
        self.save_and_restore_dictionary['summaries'] = [summary.name for summary in self.__summaries]

    @property
    def layer_structure_name(self):
        return self.__layer_structure_name

    @layer_structure_name.setter
    def layer_structure_name(self, layer_structure_name):
        self.__layer_structure_name = layer_structure_name
        self.save_and_restore_dictionary['layer_structure_name'] = self.__layer_structure_name

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
