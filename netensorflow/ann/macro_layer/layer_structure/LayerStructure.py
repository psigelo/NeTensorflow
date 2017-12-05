import random
from enum import Enum

import os

import uuid

import json

from netensorflow.ann.ANNGlobals import register_netensorflow_class, NETENSORFLOW_CLASSES


class LayerType(Enum):
    ONE_DIMENSION = 1
    IMAGE = 2


LayerTypeToString = {LayerType.ONE_DIMENSION: 'ONE_DIMENSION', LayerType.IMAGE: 'IMAGE'}
StringToLayerType = {'ONE_DIMENSION': LayerType.ONE_DIMENSION, 'IMAGE': LayerType.IMAGE}


@register_netensorflow_class
class LayerStructure(object):
    def __init__(self, layer_structure_name=None, layer_type=None, layers=None, restore=False):
        self.name = self.__class__.__name__ + '_uuid_' + uuid.uuid4().hex
        self.save_and_restore_dictionary = dict()
        self.__layer_type = None
        self.__layer_structure_name = None
        self.layers = list()

        if not restore:
            self.layer_type = layer_type
            self.layer_structure_name = layer_structure_name
            if not isinstance(layer_structure_name, str):
                raise ValueError("macro_layer_name must be string")
            if not isinstance(layer_type, LayerType):
                raise Exception('layer_type is not LayerType')

            if isinstance(layers, list):
                self.layers = layers
                for layer in layers:
                    layer.layer_type = layer_type
                    layer.layer_structure_name = layer_structure_name

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

    def save_netensorflow_model(self, path):
        layer_structure_path = os.path.join(path, self.name)
        if not os.path.exists(layer_structure_path):
            os.mkdir(layer_structure_path)

        store_dict = dict()
        store_dict['layers'] = [(ls.name, ls.__class__.__name__) for ls in self.layers]

        with open(layer_structure_path + '_data.json', 'w') as fp:
            json.dump(store_dict, fp)

        with open(layer_structure_path + '_internal_data.json', 'w') as fp:
            json.dump(self.save_and_restore_dictionary, fp)

        for layer in self.layers:
            layer.save_netensorflow_model(layer_structure_path)

    @classmethod
    def restore_netensorflow_model(cls, path, name):
        layer_structure_path = os.path.join(path, name)

        with open(layer_structure_path + '_data.json', 'r') as fp:
            layer_json_info = json.load(fp)
        layers_list = list()
        for layer_name, layer_class_name in layer_json_info['layers']:
            layer = NETENSORFLOW_CLASSES[layer_class_name]
            layers_list.append(layer.restore_netensorflow_model(layer_structure_path, layer_name))

        layer_structure = cls(restore=True)

        with open(layer_structure_path + '_internal_data.json', 'r') as fp:
            restore_json_dict = json.load(fp)

        for var_name in restore_json_dict:
            setattr(layer_structure, var_name, restore_json_dict[var_name])
        layer_structure.layers = layers_list
        layer_structure.name = name
        return layer_structure

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
    def layer_structure_name(self):
        return self.__layer_structure_name

    @layer_structure_name.setter
    def layer_structure_name(self, layer_structure_name):
        self.__layer_structure_name = layer_structure_name
        self.save_and_restore_dictionary['layer_structure_name'] = self.__layer_structure_name
