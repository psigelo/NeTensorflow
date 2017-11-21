from ann.macro_layer.layer_structure.LayerStructure import LayerStructure, LayerType
from ann.macro_layer.layer_structure.layers.InputLayer import InputLayer


class InputLayerStructure(LayerStructure):
    def __init__(self, input_dimension, dataset_dimension=None, layer_structure_name="InputLayer"):
        self.layer_type = None
        if len(input_dimension) == 4:
            self.layer_type = LayerType.IMAGE
        elif len(input_dimension) == 2:
            self.layer_type = LayerType.ONE_DIMENSION
        else:
            raise Exception('LayerType can not be deduced')
        super(InputLayerStructure, self).__init__(layer_structure_name=layer_structure_name, position=0,
                                                  layer_type=self.layer_type)
        self.__precedence_key = -1  # only for Input Layer
        self.layers = [InputLayer(input_dimension, dataset_dimension)]

        #check all layers have same layertype
        for layer in self.layers:
            if layer.layer_type != self.layer_type:
                raise Exception('LayerType is not correctly setted')
