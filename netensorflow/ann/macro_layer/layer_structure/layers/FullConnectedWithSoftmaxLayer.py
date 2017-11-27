import json

import os
import tensorflow as tf
import uuid

from netensorflow.ann.ANNGlobals import register_netensorflow_class
from netensorflow.ann.macro_layer.layer_structure.LayerStructure import LayerTypeToString, StringToLayerType
from netensorflow.ann.tensorflow_tools.variable_summaries import variable_summaries


@register_netensorflow_class
class FullConnectedWithSoftmaxLayer(object):
    def __init__(self, inputs_amount=None, restore=False):
        self.name = self.__class__.__name__ + '_uuid_' + uuid.uuid4().hex
        self.save_and_restore_dictionary = dict()
        self.__inputs_amount = None
        self.__output = None
        self.__weights = None
        self.__bias = None
        self.__layer_type = None
        self.__layer_structure_name = None
        self.__summaries = list()
        if not restore:
            self.inputs_amount = inputs_amount

    def get_tensor(self):
        if self.output is not None:
            return self.output
        else:
            raise (ValueError, "FullConnected Layer not connected, output does not exists")

    def connect_layer(self, prev_layer, input_tensor):
        with tf.name_scope('FullConnectedWithSoftmaxLayerVariables'):
            with tf.name_scope('weights'):
                self.weights = tf.Variable(tf.truncated_normal(
                    [prev_layer.inputs_amount, self.inputs_amount], stddev=0.1))
                self.summaries = self.summaries + variable_summaries(self.__weights)
            with tf.name_scope('bias'):
                self.bias = tf.Variable(tf.constant(0.1, shape=[self.inputs_amount]))
                self.summaries = self.summaries + variable_summaries(self.__bias)
            self.output = tf.nn.softmax(tf.matmul(input_tensor, self.__weights) + self.__bias)

        self.save_and_restore_dictionary.update({'weight': self.__weights.name, 'bias': self.__bias.name,
                                                 'summaries': list(map(lambda s: s.name, self.summaries))})

    def save_netensorflow_model(self, path):
        layer_path = os.path.join(path, self.name)
        with open(layer_path + '_internal_data.json', 'w') as fp:
            json.dump(self.save_and_restore_dictionary, fp)

    @property
    def layer_variables(self):
        return [self.__weights, self.__bias]

    @property
    def layer_structure_name(self):
        return self.__layer_structure_name

    @layer_structure_name.setter
    def layer_structure_name(self, layer_structure_name):
        self.__layer_structure_name = layer_structure_name
        self.save_and_restore_dictionary['layer_structure_name'] = self.__layer_structure_name

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weight):
        if isinstance(weight, str):
            weight = tf.get_default_graph().get_tensor_by_name(weight)
        self.__weights = weight
        self.save_and_restore_dictionary['weights'] = self.__weights.name

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
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, bias):
        if isinstance(bias, str):
            bias = tf.get_default_graph().get_tensor_by_name(bias)
        self.__bias = bias
        self.save_and_restore_dictionary['bias'] = self.__bias.name

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
        self.save_and_restore_dictionary['layer_name'] = LayerTypeToString[self.__layer_type]

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, output):
        if isinstance(output, str):
            output = tf.get_default_graph().get_tensor_by_name(output)
        self.__output = output
        self.save_and_restore_dictionary['output'] = self.__output.name

    def restore(self, save_and_restore_dictionary):
        self.save_and_restore_dictionary = save_and_restore_dictionary
        self.__inputs_amount = save_and_restore_dictionary['inputs_amount']
        self.__output = save_and_restore_dictionary['output']
        self.__weights = save_and_restore_dictionary['weights']
        self.__bias = save_and_restore_dictionary['bias']
        self.__layer_type = save_and_restore_dictionary['layer_type']
        self.__layer_structure_name = save_and_restore_dictionary['layer_type']
        self.__summaries = save_and_restore_dictionary['summaries']

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
