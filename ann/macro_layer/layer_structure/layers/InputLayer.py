import tensorflow as tf

from ann.macro_layer.layer_structure.LayerStructure import LayerType


class InputLayer(object):
    def __init__(self, inputs_dimension, dataset_dimension=None):
        self.inputs_amount = None
        self.filters_amount = None
        self.height_patch = None
        self.width_patch = None
        self.layer_type = None
        self.height_image = None
        self.width_image = None

        if len(inputs_dimension) == 4:
            self.layer_type = LayerType.IMAGE
            self.filters_amount = inputs_dimension[3]
            self.height_image = inputs_dimension[1]
            self.width_image = inputs_dimension[2]
        elif len(inputs_dimension) == 2:
            self.layer_type = LayerType.ONE_DIMENSION
            self.inputs_amount = inputs_dimension[1]
        else:
            raise Exception('layer_type not supported')

        dimension = dataset_dimension if dataset_dimension is not None else inputs_dimension
        with tf.name_scope('InputLayer'):
            self.inputs = tf.placeholder(tf.float32, dimension)
            self.input_reshaped = self.inputs
            if self.layer_type == LayerType.IMAGE:
                self.input_reshaped = tf.reshape(self.inputs,
                                                 [-1, self.height_image, self.width_image, self.filters_amount])

    def get_tensor(self):
        return self.input_reshaped

    @staticmethod
    def connect_layer(_):
        assert False, "Error:: Connecting process start from second layer"

    @staticmethod
    def get_variables():
        return None

    def get_input_tensor(self):
        return self.inputs
