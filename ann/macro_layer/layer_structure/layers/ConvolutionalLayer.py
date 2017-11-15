import tensorflow as tf

from ann.tensorflow_tools.variable_summaries import variable_summaries


class ConvolutionalLayer(object):
    def __init__(self, height_patch, width_patch, filters_amount, strides, padding='SAME'):
        self.prev_layer_inputs_amount = None  # must be a list of len 4 [N, H, W, C]
        self.layer_input_amount = None # must be a list of len 4 [N, H, W, C]
        self.output = None
        self.strides = strides
        self.padding = padding
        self.__weights = None
        self.__bias = None
        self.height_patch = height_patch
        self.width_patch = width_patch
        self.filters_amount = filters_amount
        self.layer_type = None
        self.height_image = None
        self.width_image = None

    def get_tensor(self):
        if self.output is not None:
            return self.output
        else:
            raise Exception("ConvolutionalLayerNotConnected")

    def connect_layer(self, prev_layer, input_tensor):
        with tf.name_scope('ConvolutionalVariables'):
            with tf.name_scope('weights'):
                self.__weights = tf.Variable(tf.truncated_normal(
                    [self.height_patch, self.width_patch, prev_layer.filters_amount, self.filters_amount],
                    stddev=0.1))
                variable_summaries(self.__weights)
            with tf.name_scope('bias'):
                self.__bias = tf.Variable(tf.constant(0.1, shape=[self.filters_amount]))
                variable_summaries(self.__bias)
            self.output = tf.nn.relu(
                tf.nn.conv2d(input_tensor, self.__weights, self.strides, padding=self.padding) + self.__bias)
            self.calc_image_height_width(prev_layer)

    def calc_image_height_width(self, prev_layer):
        if self.padding == 'VALID':
            raise Exception('Case not implemented')  # ToDo: create a algorithm to calc h and w in padding=VALID case
        elif self.padding == 'SAME':
            self.height_image = prev_layer.height_image
            self.width_image = prev_layer.width_image
        else:
            raise Exception('padding name not supported')

    def get_variables(self):
        return [self.__weights, self.__bias]
