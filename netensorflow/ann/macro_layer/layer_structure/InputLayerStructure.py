from netensorflow.ann.macro_layer.layer_structure.layers.InputLayer import InputLayer

from netensorflow.ann.macro_layer.layer_structure.LayerStructure import LayerStructure, LayerType


class InputLayerStructure(LayerStructure):
    def __init__(self, input_dimension, dataset_dimension=None, layer_structure_name="InputLayer"):
        self.save_and_restore_dictionary = dict()
        self.__layer_type = None
        if len(input_dimension) == 4:
            self.layer_type = LayerType.IMAGE
        elif len(input_dimension) == 2:
            self.layer_type = LayerType.ONE_DIMENSION
        else:
            raise Exception('LayerType can not be deduced')
        super(InputLayerStructure, self).__init__(layer_structure_name=layer_structure_name, position=0,
                                                  layer_type=self.layer_type)
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
