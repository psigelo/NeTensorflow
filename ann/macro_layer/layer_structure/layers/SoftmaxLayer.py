import tensorflow as tf


class SoftmaxLayer(object):
    def __init__(self):
        self.output = None

    def get_tensor(self, instance, owner):
        if self.output is not None:
            return self.output
        else:
            raise (ValueError, "Softmax Layer not connected, output does not exists")

    def get_input_amount(self):
        return NotImplemented

    def connect_layer(self, prev_layer_input_amount, input_tensor):
        self.output = tf.nn.softmax(input_tensor)

    @staticmethod
    def get_variables():
        return None




