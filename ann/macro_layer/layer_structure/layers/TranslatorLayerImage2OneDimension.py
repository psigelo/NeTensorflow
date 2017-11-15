import tensorflow as tf

from ann.macro_layer.layer_structure.LayerStructure import LayerType


class TranslatorLayerImage2OneDimesion(object):
    def __init__(self):
        self.inputs_amount = None
        self.output = None

    def get_tensor(self):
        if self.output is not None:
            return self.output
        else:
            raise (ValueError, "FullConnected Layer not connected, output does not exists")

    def connect_layer(self, prev_layer, input_tensor):
        if prev_layer.layer_type != LayerType.IMAGE:
            raise Exception('PrevLayerMustBeTypeImage')

        self.inputs_amount = prev_layer.height_image * prev_layer.width_image * prev_layer.filters_amount
        self.output = tf.reshape(input_tensor, [-1, self.inputs_amount])
