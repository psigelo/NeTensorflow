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

    def get_input_amount(self):
        if self.input_amount is None:
            raise Exception('You must connect this layer before ask input amount because is the same as the previous')
        return self.input_amount

    def connect_layer(self, input_amount, input_tensor):
        self.output = tf.nn.softmax(input_tensor)
        self.input_amount = input_amount

    @staticmethod
    def get_variables():
        return None
