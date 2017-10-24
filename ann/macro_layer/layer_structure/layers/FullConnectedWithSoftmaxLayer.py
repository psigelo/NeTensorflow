import tensorflow as tf


class FullConnectedWithSoftmaxLayer(object):
    def __init__(self, inputs_amount=None):
        self.inputs_amount = inputs_amount
        self.output = None
        self.__weights = None
        self.__bias = None

    def get_tensor(self):
        if self.output is not None:
            return self.output
        else:
            raise (ValueError, "FullConnectedWithSoftmaxLayer Layer not connected, output does not exists")

    def get_input_amount(self):
        return self.inputs_amount

    def connect_layer(self, prev_layer_input_amount, input_tensor):
        if isinstance(prev_layer_input_amount, list):
            if len(prev_layer_input_amount) > 1:
                raise (NotImplemented, "case not implemented:: input_size is list larger than 1 item")
        elif isinstance(prev_layer_input_amount, int):
            self.__weights = tf.Variable(tf.truncated_normal([prev_layer_input_amount, self.inputs_amount], stddev=0.1))
            self.__bias = tf.Variable(tf.constant(0.1, shape=[self.inputs_amount]))
            self.output = tf.nn.softmax(tf.matmul(input_tensor, self.__weights) + self.__bias)
        else:
            raise (NotImplemented, "input size type not recognized")

    def get_variables(self):
        return [self.__weights, self.__bias]
