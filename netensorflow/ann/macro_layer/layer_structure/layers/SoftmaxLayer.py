import json

import os
import tensorflow as tf
import uuid

from netensorflow.ann.ANNGlobals import register_netensorflow_class
from netensorflow.ann.macro_layer.layer_structure.LayerStructure import LayerTypeToString, StringToLayerType
from netensorflow.ann.macro_layer.layer_structure.layers.AbsctractLayer import AbstractLayer


@register_netensorflow_class
class SoftmaxLayer(AbstractLayer):
    def __init__(self, restore=False):
        super(SoftmaxLayer, self).__init__()
        self.name = self.__class__.__name__ + '_uuid_' + uuid.uuid4().hex
        self.save_and_restore_dictionary = dict()
        self.__output = None
        self.__inputs_amount = None
        self.__layer_type = None
        self.__layer_structure_name = None

    def get_tensor(self):
        if self.output is not None:
            return self.output
        else:
            raise (ValueError, "Softmax Layer not connected, output does not exists")

    def connect_layer(self, prev_layer, input_tensor):
        self.output = tf.nn.softmax(input_tensor)
        self.inputs_amount = prev_layer.inputs_amount

    def save_netensorflow_model(self, path):
        layer_path = os.path.join(path, self.name)
        with open(layer_path + '_internal_data.json', 'w') as fp:
            json.dump(self.save_and_restore_dictionary, fp)

    @property
    def layer_variables(self):
        return list()

    @staticmethod
    def get_variables():
        return None

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
    def layer_type(self):
        return self.__layer_type

    @layer_type.setter
    def layer_type(self, layer_type):
        if isinstance(layer_type, str):
            layer_type = StringToLayerType[layer_type]
        self.__layer_type = layer_type
        self.save_and_restore_dictionary['layer_type'] = LayerTypeToString[self.__layer_type]

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
