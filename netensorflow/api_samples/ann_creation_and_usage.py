import tensorflow as tf

from netensorflow.ann.ANN import ANN
from netensorflow.ann.macro_layer.MacroLayer import MacroLayer
from netensorflow.ann.macro_layer.layer_structure.InputLayerStructure import InputLayerStructure
from netensorflow.ann.macro_layer.layer_structure.LayerStructure import LayerStructure, LayerType
from netensorflow.ann.macro_layer.layer_structure.layers.FullConnected import FullConnected
from netensorflow.ann.macro_layer.layer_structure.layers.FullConnectedWithSoftmaxLayer import FullConnectedWithSoftmaxLayer

'''
    ann Creation and simple usage, the goal of this code is simply run the most simpler artificial neural network 

'''


def main():
    # tensorflow
    tf_sess = tf.Session()

    # Layers:
    input_dim = [None, 3]
    hidden_layer = FullConnected(inputs_amount=20)
    out_layer = FullConnectedWithSoftmaxLayer(inputs_amount=10)

    # Layer Structures
    input_layer_structure = InputLayerStructure(input_dim)
    hidden_layer_structure = LayerStructure('Hidden', position=0, layer_type=LayerType.ONE_DIMENSION, layers=[hidden_layer])
    output_layer_structure = LayerStructure('Output', position=1, layer_type=LayerType.ONE_DIMENSION,layers=[out_layer])

    # Macro Layer
    macro_layers = MacroLayer(layers_structure=[input_layer_structure, hidden_layer_structure, output_layer_structure])

    # ann
    ann = ANN(macro_layers=macro_layers, tf_session=tf_sess, base_folder='./tensorboard_logs/')
    ann.connect_and_initialize()

    # Execute
    for it in range(100):
        import numpy as np
        input_tensor_value = [np.random.uniform(0.0, 10.0, 3)]
        print(ann.run(global_iteration=it, input_tensor_value=input_tensor_value))


if __name__ == '__main__':
    main()
