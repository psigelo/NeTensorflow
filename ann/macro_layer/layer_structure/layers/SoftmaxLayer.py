import tensorflow as tf


class SoftmaxLayer(object):
    def __init__(self):
        self.output = None

    def get_tensor(self):
        if self.output is not None:
            return self.output
        else:
            raise (ValueError, "Softmax Layer not connected, output does not exists")

    @staticmethod
    def get_input_amount():
        return NotImplemented

    def connect_layer(self, _, input_tensor):
        self.output = tf.nn.softmax(input_tensor)

    @staticmethod
    def get_variables():
        return None
