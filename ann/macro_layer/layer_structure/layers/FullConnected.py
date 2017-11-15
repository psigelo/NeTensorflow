import tensorflow as tf

from ann.tensorflow_tools.variable_summaries import variable_summaries


class FullConnected(object):
    def __init__(self, inputs_amount=None):
        self.inputs_amount = inputs_amount
        self.output = None
        self.__weights = None
        self.__bias = None
        self.layer_type = None

    def get_tensor(self):
        if self.output is not None:
            return self.output
        else:
            raise (ValueError, "FullConnected Layer not connected, output does not exists")

    def connect_layer(self, prev_layer, input_tensor):
        with tf.name_scope('FullConnectedVariables'):
            with tf.name_scope('weights'):
                self.__weights = tf.Variable(tf.truncated_normal(
                    [prev_layer.inputs_amount, self.inputs_amount], stddev=0.1))
                variable_summaries(self.__weights)
            with tf.name_scope('bias'):
                self.__bias = tf.Variable(tf.constant(0.1, shape=[self.inputs_amount]))
                variable_summaries(self.__bias)
            self.output = tf.matmul(input_tensor, self.__weights) + self.__bias

    def get_variables(self):
        return [self.__weights, self.__bias]
