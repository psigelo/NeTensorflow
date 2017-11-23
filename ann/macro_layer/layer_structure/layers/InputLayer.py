import tensorflow as tf

from ann.macro_layer.layer_structure.LayerStructure import LayerType


class InputLayer(object):
    def __init__(self, inputs_dimension, dataset_dimension=None):
        self.save_and_restore_dictionary = dict()
        self.__inputs_amount = None
        self.__filters_amount = None
        self.__height_patch = None
        self.__width_patch = None
        self.__layer_type = None
        self.__height_image = None
        self.__width_image = None
        self.__layer_structure_name = None
        self.__summaries = list()

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

    @property
    def layer_variables(self):
        return list()

    def get_input_tensor(self):
        return self.inputs

    @property
    def inputs_amount(self):
        return self.__inputs_amount

    @inputs_amount.setter
    def inputs_amount(self, inputs_amount):
        self.__inputs_amount = inputs_amount
        self.save_and_restore_dictionary['inputs_amount'] = self.__inputs_amount

    @property
    def filters_amount(self):
        return self.__filters_amount

    @filters_amount.setter
    def filters_amount(self, filters_amount):
        self.__filters_amount = filters_amount
        self.save_and_restore_dictionary['filters_amount'] = self.__filters_amount

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
