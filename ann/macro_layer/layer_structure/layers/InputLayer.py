import tensorflow as tf


class InputLayer(object):
    def __init__(self, inputs_amount):
        if isinstance(inputs_amount, int):
            self.inputs_amount = inputs_amount

        elif isinstance(inputs_amount, list):
            if inputs_amount[0] is None:
                self.inputs_amount = inputs_amount[1]
            else:
                raise Exception('NotImplemented', 'No yet implemented input case')
        else:
            raise Exception('NotImplemented', 'No yet implemented input case')

        with tf.name_scope('Input'):
            self.inputs = tf.placeholder(tf.float32, [None, self.inputs_amount])

    def get_tensor(self):
        return self.inputs

    def get_input_amount(self):
        return self.inputs_amount

    @staticmethod
    def connect_layer(_):
        assert False, "Error:: Connecting process start from second layer"

    @staticmethod
    def get_variables():
        return None

    def get_input_tensor(self):
        return self.inputs
