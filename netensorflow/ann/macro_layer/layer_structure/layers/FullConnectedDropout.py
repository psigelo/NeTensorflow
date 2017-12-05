import tensorflow as tf

from netensorflow.ann.ANNGlobals import register_netensorflow_class
from netensorflow.ann.macro_layer.layer_structure.layers import FullConnected


@register_netensorflow_class
class FullConnectedDropout(FullConnected):
    def __init__(self, inputs_amount=None, keep_prob=None, restore=False):
        self.__drop_output = None
        self.__keep_prob = None
        if restore:
            super(FullConnectedDropout, self).__init__(restore=True)
        else:
            super(FullConnectedDropout, self).__init__(inputs_amount=inputs_amount, restore=restore)
            self.__keep_prob = keep_prob

    def connect_layer(self, prev_layer, input_tensor):
        with tf.name_scope('Dropout'):
            super(FullConnectedDropout, self).connect_layer(prev_layer, input_tensor)
            self.drop_output = tf.nn.dropout(self.output, self.keep_prob)

    def get_placeholder(self):
        return self.keep_prob

    def get_tensor(self):
        return self.__drop_output

    @property
    def drop_output(self):
        return self.__drop_output

    @drop_output.setter
    def drop_output(self, drop_output):
        if isinstance(drop_output, str):
            drop_output = tf.get_default_graph().get_tensor_by_name(drop_output)
        self.__drop_output = drop_output
        self.save_and_restore_dictionary['drop_output'] = self.__drop_output.name

    @property
    def keep_prob(self):
        return self.__keep_prob

    @keep_prob.setter
    def keep_prob(self, keep_prob):
        self.__keep_prob = keep_prob
        self.save_and_restore_dictionary['keep_prob'] = self.__keep_prob
