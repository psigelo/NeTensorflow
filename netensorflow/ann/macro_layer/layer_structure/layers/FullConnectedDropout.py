import tensorflow as tf

from netensorflow.ann.ANNGlobals import register_netensorflow_class
from netensorflow.ann.macro_layer.layer_structure.layers import FullConnected


@register_netensorflow_class
class FullConnectedDropout(FullConnected):
    def __init__(self, inputs_amount=None, keep_prob=None, restore=False):
        self.__drop_output = None
        self.__keep_prob_feed = None
        self.__keep_prob = None
        if restore:
            super(FullConnectedDropout, self).__init__(restore=True)
        else:
            super(FullConnectedDropout, self).__init__(inputs_amount=inputs_amount, restore=restore)
            self.__keep_prob = keep_prob

    def connect_layer(self, prev_layer, input_tensor):
        with tf.name_scope('Dropout'):
            super(FullConnectedDropout, self).connect_layer(prev_layer, input_tensor)
            with tf.device('/cpu:0'):
                self.keep_prob_feed = tf.placeholder(tf.float32)
            self.drop_output = tf.nn.dropout(self.output, self.__keep_prob_feed)

    def get_tensor(self):
        return self.__drop_output

    @property
    def layer_hidden_placeholder(self):
        return {'placeholder': self.keep_prob_feed, 'training': self.keep_prob, 'running': 1.0}

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

    @property
    def keep_prob_feed(self):
        return self.__keep_prob_feed

    @keep_prob_feed.setter
    def keep_prob_feed(self, keep_prob_feed):
        if isinstance(keep_prob_feed, str):
            keep_prob_feed = tf.get_default_graph().get_tensor_by_name(keep_prob_feed)
        self.__keep_prob_feed = keep_prob_feed
        self.save_and_restore_dictionary['keep_prob_feed'] = self.__keep_prob_feed.name
