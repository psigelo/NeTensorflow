import os

import json

import uuid

from netensorflow.ann.ANNGlobals import register_netensorflow_class, NETENSORFLOW_CLASSES
from netensorflow.ann.macro_layer.layer_structure.layers.InputLayer import InputLayer
from netensorflow.ann.macro_layer.layer_structure.LayerStructure import LayerStructure, LayerType, StringToLayerType, \
    LayerTypeToString


@register_netensorflow_class
class InputLayerStructure(LayerStructure):
    def __init__(self, input_dimension=None, layer_structure_name="InputLayer", restore=False,
                 layer_type=None):
        if not restore:
            super(InputLayerStructure, self).__init__(layer_structure_name=layer_structure_name,
                                                      layer_type=layer_type)
            self.layers = [InputLayer(input_dimension, layer_type=layer_type)]
        else:
            super(InputLayerStructure, self).__init__(restore=True)
