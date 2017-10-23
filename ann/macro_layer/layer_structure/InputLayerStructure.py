import tensorflow as tf

from ann.macro_layer.layer_structure.LayerStructure import LayerStructure
from ann.macro_layer.layer_structure.layers.InputLayer import InputLayer


class InputLayerStructure(LayerStructure):
    def  __init__(self, input_dimension, macro_layer_name="InputLayer"):
        super(InputLayerStructure, self).__init__(macro_layer_name=macro_layer_name, position=0)
        self.__precedence_key = -1  # only for Input Layer
        if not isinstance(input_dimension, list):
            raise (ValueError, "input dimension must be a list")
        self.layers = [InputLayer(input_dimension)]
