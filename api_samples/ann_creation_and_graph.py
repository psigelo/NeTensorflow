import tensorflow as tf

from ann.ANN import ANN
from ann.macro_layer.MacroLayer import MacroLayer
from ann.macro_layer.layer_structure.InputLayerStructure import InputLayerStructure
from ann.macro_layer.layer_structure.LayerStructure import LayerStructure
from ann.macro_layer.layer_structure.layers.FullConnected import FullConnected
from ann.macro_layer.layer_structure.layers.FullConnectedWithSoftmaxLayer import FullConnectedWithSoftmaxLayer

'''
    ann Creation and simple usage, the goal of this code is simply run the most simpler artificial neural network 

'''


def main():
    # tensorflow_tools
    tf_sess = tf.Session()

    # Layers:
    input_dim = [None, 3]
    hidden_layer = FullConnected(inputs_amount=20)
    out_layer = FullConnectedWithSoftmaxLayer(inputs_amount=10)

    # Layer Structures
    input_layer_structure = InputLayerStructure(input_dim)
    hidden_layer_structure = LayerStructure('Hidden', position=0, layers=[hidden_layer])
    output_layer_structure = LayerStructure('Output', position=1, layers=[out_layer])

    # Macro Layer
    macro_layers = MacroLayer(layers_structure=[input_layer_structure, hidden_layer_structure, output_layer_structure])

    # ann
    ann = ANN(macro_layers=macro_layers, tf_session=tf_sess, base_folder='./graphs_created/')
    ann.connect_and_initialize()

    # Create the graph
    ann.write_graph_and_summaries(global_iteration=0)


if __name__ == '__main__':
    main()
