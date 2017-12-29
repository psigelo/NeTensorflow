import os

import json
import tensorflow as tf
import uuid

from netensorflow.ann.ANNGlobals import register_netensorflow_class
from netensorflow.ann.macro_layer.layer_structure.LayerStructure import LayerType, LayerTypeToString, StringToLayerType
from netensorflow.ann.macro_layer.layer_structure.layers.AbsctractLayer import AbstractLayer


@register_netensorflow_class
class InputLayer(AbstractLayer):
    def __init__(self, inputs_dimension=None, layer_type=None, restore=False):
        super(AbstractLayer, self).__init__()
        self.name = self.__class__.__name__ + '_uuid_' + uuid.uuid4().hex
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
        self.__inputs = None
        self.__inputs_dimension = None
        self.__dataset_dimension = None

        if not restore:
            self.layer_type = layer_type
            self.inputs_dimension = inputs_dimension
            if len(inputs_dimension) == 4:
                self.filters_amount = inputs_dimension[3]
                self.height_image = inputs_dimension[1]
                self.width_image = inputs_dimension[2]
            elif len(inputs_dimension) == 2:
                self.inputs_amount = inputs_dimension[1]
            else:
                raise Exception('layer_type not supported')

            with tf.name_scope('InputLayer'):
                self.inputs = tf.placeholder(tf.float32, inputs_dimension)

    def get_tensor(self):
        return self.inputs

    def save_netensorflow_model(self, path):
        layer_path = os.path.join(path, self.name)
        with open(layer_path + '_internal_data.json', 'w') as fp:
            json.dump(self.save_and_restore_dictionary, fp)

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
    def dataset_dimension(self):
        return self.__dataset_dimension

    @dataset_dimension.setter
    def dataset_dimension(self, dataset_dimension):
        self.__dataset_dimension = dataset_dimension
        self.save_and_restore_dictionary['dataset_dimension'] = self.__dataset_dimension

    @property
    def inputs_dimension(self):
        return self.__inputs_dimension

    @inputs_dimension.setter
    def inputs_dimension(self, inputs_dimension):
        self.__inputs_dimension = inputs_dimension
        self.save_and_restore_dictionary['inputs_dimension'] = self.__inputs_dimension

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
        if isinstance(layer_type, str):
            layer_type = StringToLayerType[layer_type]
        self.__layer_type = layer_type
        self.save_and_restore_dictionary['layer_type'] = LayerTypeToString[self.__layer_type]

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
        summaries_ = None
        if len(summaries) > 0:
            if isinstance(summaries[0], str):  # then is restoring, and is in string format.
                summaries_ = [tf.get_default_graph().get_tensor_by_name(summary) for summary in summaries]
        if summaries_ is None:
            summaries_ = summaries
        self.__summaries = summaries_
        self.save_and_restore_dictionary['summaries'] = [summary.name for summary in self.__summaries]

    @property
    def inputs(self):
        return self.__inputs

    @inputs.setter
    def inputs(self, inputs):
        if isinstance(inputs, str):
            inputs = tf.get_default_graph().get_tensor_by_name(inputs)  # we suppose that is being restore
        self.__inputs = inputs
        self.save_and_restore_dictionary['inputs'] = self.__inputs.name

    @classmethod
    def restore_netensorflow_model(cls, path, name):
        layer_path = os.path.join(path, name)
        with open(layer_path + '_internal_data.json', 'r') as fp:
            restore_json_dict = json.load(fp)

        layer = cls(restore=True)
        for var_name in restore_json_dict:
            setattr(layer, var_name, restore_json_dict[var_name])
        layer.name = name
        return layer
