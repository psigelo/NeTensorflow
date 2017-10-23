import tensorflow as tf


class InputLayer(object):
    def __init__(self, inputs_amount):
        self.inputs_amount = inputs_amount
        with tf.name_scope('Input'):
            self.inputs = tf.placeholder(tf.float32, inputs_amount)

    def get_tensor(self, instance, owner):
        return self.inputs

    def get_input_amount(self):
        return self.inputs_amount

    def connect_layer(self, prev_layer_input_amount, input_tensor):
        assert False, "Error:: Connecting process start from second layer"

    def get_variables(self):
        return None

    def get_input_tensor(self):
        return self.inputs




