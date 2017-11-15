import tensorflow as tf


class SoftmaxLayer(object):
    def __init__(self):
        self.output = None
        self.input_amount = None

    def get_tensor(self):
        if self.output is not None:
            return self.output
        else:
            raise (ValueError, "Softmax Layer not connected, output does not exists")

    def connect_layer(self, prev_layer, input_tensor):
        self.output = tf.nn.softmax(input_tensor)
        self.inputs_amount = prev_layer.inputs_amount

    @staticmethod
    def get_variables():
        return None
