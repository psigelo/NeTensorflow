import os

import json

import uuid

from netensorflow.ann.ANNGlobals import register_netensorflow_class
from netensorflow.ann.macro_layer.layer_structure.layers.InputLayer import InputLayer

from netensorflow.ann.macro_layer.layer_structure.LayerStructure import LayerStructure, LayerType


@register_netensorflow_class
class InputLayerStructure(LayerStructure):
    def __init__(self, input_dimension, dataset_dimension=None, layer_structure_name="InputLayer"):
        self.name = self.__class__.__name__ + '_uuid_' + uuid.uuid4().hex
        self.save_and_restore_dictionary = dict()
        self.__layer_type = None
        if len(input_dimension) == 4:
            self.layer_type = LayerType.IMAGE
        elif len(input_dimension) == 2:
            self.layer_type = LayerType.ONE_DIMENSION
        else:
            raise Exception('LayerType can not be deduced')
        super(InputLayerStructure, self).__init__(layer_structure_name=layer_structure_name, layer_type=self.layer_type)
        self.layers = [InputLayer(input_dimension, dataset_dimension)]

        # check all layers have same layer type
        for layer in self.layers:
            if layer.layer_type != self.layer_type:
                raise Exception('LayerType is not correctly setted')

    @property
    def layer_type(self):
        return self.__layer_type

    @layer_type.setter
    def layer_type(self, layer_type):
        self.__layer_type = layer_type
        self.save_and_restore_dictionary['layer_type'] = self.__layer_type

    def save_netensorflow_model(self, path):
        layer_structure_path = os.path.join(path, self.name)
        if not os.path.exists(layer_structure_path):
            os.mkdir(layer_structure_path)

        store_dict = dict()
        store_dict['layers'] = [(ls.name, ls.__class__.__name__) for ls in self.layers]

        with open(layer_structure_path + '_data.json', 'w') as fp:
            json.dump(store_dict, fp)

        for layer in self.layers:
            layer.save_netensorflow_model(layer_structure_path)
