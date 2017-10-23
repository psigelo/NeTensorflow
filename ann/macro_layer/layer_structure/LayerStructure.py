import random


class LayerStructure(object):
    def __init__(self, macro_layer_name, position, layers=None):
        if not isinstance(macro_layer_name, str):
            raise ValueError("macro_layer_name must be string")
        if isinstance(position, int):
            if position < 0:
                raise(ValueError, "position must be a integer greater or equal than 0")
        self.macro_layer_name = macro_layer_name
        self.__precedence_key = position
        self.layers = list()

        if isinstance(layers, list):
            self.layers = layers

    @property
    def precedence_key(self):
        return self.__precedence_key

    @precedence_key.setter
    def precedence_key(self, precedence_key):
        if precedence_key < 0:
            raise(ValueError, "position must be a integer greater or equal than 0")
        self.__precedence_key = precedence_key

    def add_layer(self, layer, layer_position_place=None):
        if isinstance(layer_position_place, int):
            if (layer_position_place < len(layer)) and (layer_position_place >= 0):
                position_new_layer = layer_position_place
            else:
                raise (ValueError, "layer_position_place must out of space")
        else:
            before_layers_amount = len(layer)
            position_new_layer = random.randint(0, before_layers_amount)
        self.layers.insert(position_new_layer, layer)

