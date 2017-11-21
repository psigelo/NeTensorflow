import tensorflow as tf


class SoftmaxLayer(object):
    def __init__(self):
        self.output = None
        self.inputs_amount = None
        self.layer_type = None
        self.layer_structure_name = None

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
