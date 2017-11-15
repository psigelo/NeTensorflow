import random
from enum import Enum


class LayerType(Enum):
    ONE_DIMENSION = 1
    IMAGE = 2


class LayerStructure(object):
    def __init__(self, macro_layer_name, position, layer_type, layers=None):
        if not isinstance(macro_layer_name, str):
            raise ValueError("macro_layer_name must be string")
        if isinstance(position, int):
            if position < 0:
                raise(ValueError, "position must be a integer greater or equal than 0")
        if not isinstance(layer_type, LayerType):
            raise Exception('layer_type is not LayerType')
        self.layer_type = layer_type
        self.macro_layer_name = macro_layer_name
        self.__precedence_key = position
        self.layers = list()

        if isinstance(layers, list):
            self.layers = layers
            for layer in layers:
                layer.layer_type = layer_type

    @property
    def precedence_key(self):
        return self.__precedence_key

    @precedence_key.setter
    def precedence_key(self, precedence_key):
        if precedence_key < 0:
            raise(ValueError, "position must be a integer greater or equal than 0")
        self.__precedence_key = precedence_key

    def add_layer(self, layer, layer_position_place=None):
        layer.layer_type = self.layer_type
        if isinstance(layer_position_place, int):
            if (layer_position_place < len(layer)) and (layer_position_place >= 0):
                position_new_layer = layer_position_place
            else:
                raise (ValueError, "layer_position_place must out of space")
        else:
            before_layers_amount = len(layer)
            position_new_layer = random.randint(0, before_layers_amount)
        self.layers.insert(position_new_layer, layer)
