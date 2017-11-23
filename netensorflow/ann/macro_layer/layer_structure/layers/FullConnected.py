import tensorflow as tf

from netensorflow.ann.tensorflow_tools.variable_summaries import variable_summaries


class FullConnected(object):
    def __init__(self, inputs_amount=None):
        self.save_and_restore_dictionary = dict()
        self.__inputs_amount = None
        self.__output = None
        self.__weights = None
        self.__bias = None
        self.__layer_type = None
        self.__layer_structure_name = None
        self.__summaries = list()
        self.inputs_amount = inputs_amount

    def get_tensor(self):
        if self.output is not None:
            return self.output
        else:
            raise (ValueError, "FullConnected Layer not connected, output does not exists")

    def connect_layer(self, prev_layer, input_tensor):
        with tf.name_scope('FullConnectedVariables'):
            with tf.name_scope('weights'):
                self.weights = tf.Variable(tf.truncated_normal(
                    [prev_layer.inputs_amount, self.inputs_amount], stddev=0.1))
#                self.summaries = self.summaries + variable_summaries(self.weights)
            with tf.name_scope('bias'):
                self.bias = tf.Variable(tf.constant(0.1, shape=[self.inputs_amount]))
                self.summaries = self.summaries + variable_summaries(self.bias)
            self.output = tf.matmul(input_tensor, self.weights) + self.bias

        self.save_and_restore_dictionary.update({'weight': self.weights.name, 'bias': self.bias.name,
                                                 'summaries': list(map(lambda s: s.name, self.summaries))})

    @property
    def layer_variables(self):
        return [self.__weights, self.__bias]

    @property
    def layer_structure_name(self):
        return self.__layer_structure_name

    @layer_structure_name.setter
    def layer_structure_name(self, layer_structure_name):
        self.__layer_structure_name = layer_structure_name
        self.save_and_restore_dictionary['layer_structure_name'] = self.__layer_type

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weight):
        self.__weights = weight
        self.save_and_restore_dictionary['weights'] = self.__weights

    @property
    def summaries(self):
        return self.__summaries

    @summaries.setter
    def summaries(self, summaries):
        self.__summaries = summaries
        self.save_and_restore_dictionary['summaries'] = self.__summaries

    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, bias):
        self.__bias = bias
        self.save_and_restore_dictionary['bias'] = self.__bias

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
    def layer_type(self, layer_name):
        self.__layer_type = layer_name
        self.save_and_restore_dictionary['layer_name'] = self.__layer_type

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, output):
        self.__output = output
        self.save_and_restore_dictionary['output'] = self.__output

    def restore(self, save_and_restore_dictionary):
        self.save_and_restore_dictionary = save_and_restore_dictionary
        self.__inputs_amount = save_and_restore_dictionary['inputs_amount']
        self.__output = save_and_restore_dictionary['output']
        self.__weights = save_and_restore_dictionary['weights']
        self.__bias = save_and_restore_dictionary['bias']
        self.__layer_type = save_and_restore_dictionary['layer_type']
        self.__layer_structure_name = save_and_restore_dictionary['layer_type']
        self.__summaries = save_and_restore_dictionary['summaries']