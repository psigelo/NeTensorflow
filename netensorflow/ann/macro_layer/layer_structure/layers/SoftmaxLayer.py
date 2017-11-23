import tensorflow as tf


class SoftmaxLayer(object):
    def __init__(self):
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
        self.__output = output
        self.save_and_restore_dictionary['output'] = self.__output

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
        self.__layer_type = layer_type
        self.save_and_restore_dictionary['layer_type'] = self.__layer_type

    @property
    def layer_structure_name(self):
        return self.__layer_structure_name

    @layer_structure_name.setter
    def layer_structure_name(self, layer_structure_name):
        self.__layer_structure_name = layer_structure_name
        self.save_and_restore_dictionary['layer_structure_name'] = self.__layer_structure_name
