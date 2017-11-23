import tensorflow as tf

from ann.tensorflow_tools.variable_summaries import variable_summaries


class ConvolutionalLayer(object):
    def __init__(self, height_patch, width_patch, filters_amount, strides, padding='SAME'):
        self.save_and_restore_dictionary = dict()
        self.__prev_layer_inputs_amount = None  # must be a list of len 4 [N, H, W, C]
        self.__layer_input_amount = None  # must be a list of len 4 [N, H, W, C]
        self.__output = None
        self.__strides = None
        self.__padding = None
        self.__weights = None
        self.__bias = None
        self.__height_patch = None
        self.__width_patch = None
        self.__filters_amount = None
        self.__layer_type = None
        self.__height_image = None
        self.__width_image = None
        self.__layer_structure_name = None
        self.__summaries = list()
        self.strides = strides
        self.padding = padding
        self.height_patch = height_patch
        self.width_patch = width_patch
        self.filters_amount = filters_amount

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
                self.summaries = self.summaries + variable_summaries(self.__weights)
            with tf.name_scope('bias'):
                self.__bias = tf.Variable(tf.constant(0.1, shape=[self.filters_amount]))
                self.summaries = self.summaries + variable_summaries(self.__bias)
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

    @property
    def layer_variables(self):
        return [self.__weights, self.__bias]

    @property
    def prev_layer_inputs_amount(self):
        return self.__prev_layer_inputs_amount

    @prev_layer_inputs_amount.setter
    def prev_layer_inputs_amount(self, prev_layer_inputs_amount):
        self.__prev_layer_inputs_amount = prev_layer_inputs_amount
        self.save_and_restore_dictionary['prev_layer_inputs_amount'] = self.__prev_layer_inputs_amount

    @property
    def layer_input_amount(self):
        return self.__layer_input_amount

    @layer_input_amount.setter
    def layer_input_amount(self, layer_input_amount):
        self.__layer_input_amount = layer_input_amount
        self.save_and_restore_dictionary['layer_input_amount'] = self.__layer_input_amount

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, output):
        self.__output = output
        self.save_and_restore_dictionary['output'] = self.__output

    @property
    def strides(self):
        return self.__strides

    @strides.setter
    def strides(self, strides):
        self.__strides = strides
        self.save_and_restore_dictionary['strides'] = self.__strides

    @property
    def padding(self):
        return self.__padding

    @padding.setter
    def padding(self, padding):
        self.__padding = padding
        self.save_and_restore_dictionary['padding'] = self.__padding

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights):
        self.__weights = weights
        self.save_and_restore_dictionary['weights'] = self.__weights

    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, bias):
        self.__bias = bias
        self.save_and_restore_dictionary['bias'] = self.__bias

    @property
    def height_patch(self):
        return self.__height_patch

    @height_patch.setter
    def height_patch(self, height_patch):
        self.__height_patch = height_patch
        self.save_and_restore_dictionary['height_patch'] = self.__height_patch

    @property
    def width_patch(self):
        return self.__width_patch

    @width_patch.setter
    def width_patch(self, width_patch):
        self.__width_patch = width_patch
        self.save_and_restore_dictionary['width_patch'] = self.__width_patch

    @property
    def filters_amount(self):
        return self.__filters_amount

    @filters_amount.setter
    def filters_amount(self, filters_amount):
        self.__filters_amount = filters_amount
        self.save_and_restore_dictionary['filters_amount'] = self.__filters_amount

    @property
    def layer_type(self):
        return self.__layer_type

    @layer_type.setter
    def layer_type(self, layer_type):
        self.__layer_type = layer_type
        self.save_and_restore_dictionary['layer_type'] = self.__layer_type

    @property
    def height_image(self):
        return self.__height_image

    @height_image.setter
    def height_image(self, height_image):
        self.__height_image = height_image
        self.save_and_restore_dictionary['height_image'] = self.__height_image

    @property
    def width_image(self):
        return self.__width_image

    @width_image.setter
    def width_image(self, width_image):
        self.__width_image = width_image
        self.save_and_restore_dictionary['width_image'] = self.__width_image

    @property
    def layer_structure_name(self):
        return self.__layer_structure_name

    @layer_structure_name.setter
    def layer_structure_name(self, layer_structure_name):
        self.__layer_structure_name = layer_structure_name
        self.save_and_restore_dictionary['layer_structure_name'] = self.__layer_structure_name

    @property
    def summaries(self):
        return self.__summaries

    @summaries.setter
    def summaries(self, summaries):
        self.__summaries = summaries
        self.save_and_restore_dictionary['summaries'] = self.__summaries
