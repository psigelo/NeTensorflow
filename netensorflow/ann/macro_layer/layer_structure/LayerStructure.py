import random
from enum import Enum

import os

import uuid

import json


class LayerType(Enum):
    ONE_DIMENSION = 1
    IMAGE = 2


LayerTypeToString = {LayerType.ONE_DIMENSION: 'ONE_DIMENSION', LayerType.IMAGE: 'IMAGE'}
StringToLayerType = {'ONE_DIMENSION': LayerType.ONE_DIMENSION, 'IMAGE': LayerType.IMAGE}


class LayerStructure(object):
    def __init__(self, layer_structure_name, position, layer_type, layers=None):
        self.name = self.__class__.__name__ + '_uuid_' + uuid.uuid4().hex
        if not isinstance(layer_structure_name, str):
            raise ValueError("macro_layer_name must be string")
        if isinstance(position, int):
            if position < 0:
                raise(ValueError, "position must be a integer greater or equal than 0")
        if not isinstance(layer_type, LayerType):
            raise Exception('layer_type is not LayerType')
        self.layer_type = layer_type
        self.layer_structure_name = layer_structure_name
        self.__precedence_key = position
        self.layers = list()

        if isinstance(layers, list):
            self.layers = layers
            for layer in layers:
                layer.layer_type = layer_type
                layer.layer_structure_name = layer_structure_name

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

    def add_layer_at_bottom(self, layer):
        self.layers.insert(0, layer)

    def add_layer_at_top(self, layer):
        pass

    def save_netensorflow_model(self, path):
        layer_structure_path = os.path.join(path, self.name)
        if not os.path.exists(layer_structure_path):
            os.mkdir(layer_structure_path)

        store_dict = dict()
        store_dict['layers'] = [(ls.name, ls.__class__.__name__) for ls in self.layers]

        with open(layer_structure_path + 'data.json', 'w') as fp:
            json.dump(store_dict, fp)

        for layer in self.layers:
            layer.save_netensorflow_model(layer_structure_path)
