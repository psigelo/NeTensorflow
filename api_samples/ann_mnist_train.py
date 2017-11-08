import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

from ann.ANN import ANN
from ann.macro_layer.MacroLayer import MacroLayer
from ann.macro_layer.layer_structure.InputLayerStructure import InputLayerStructure
from ann.macro_layer.layer_structure.LayerStructure import LayerStructure
from ann.macro_layer.layer_structure.layers.FullConnected import FullConnected
from ann.macro_layer.layer_structure.layers.FullConnectedWithSoftmaxLayer import FullConnectedWithSoftmaxLayer
from ann.macro_layer.trainers.DefaultTrainer import DefaultTrainer

'''
    ANN mnist training simple sample

'''


def main():
    # Input / Output
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Tensorflow
    tf_sess = tf.Session()

    # Layers
    input_dim = [None, 784]
    hidden_layer = FullConnected(inputs_amount=300)
    out_layer = FullConnectedWithSoftmaxLayer(inputs_amount=10)

    # Layer Structures
    input_layer_structure = InputLayerStructure(input_dim)
    hidden_layer_structure = LayerStructure('Hidden', position=0, layers=[hidden_layer])
    output_layer_structure = LayerStructure('Output', position=1, layers=[out_layer])

    # Macro Layer
    macro_layers = MacroLayer(layers_structure=[input_layer_structure, hidden_layer_structure, output_layer_structure])

    # Train
    trainer = DefaultTrainer(layers_structures=[input_layer_structure, hidden_layer_structure, output_layer_structure])

    # ANN
    ann = ANN(macro_layers=macro_layers, tf_session=tf_sess, base_folder='./tensorboard_logs/', trainer_list=[trainer])
    ann.connect_and_initialize()

    # Execute
    for it in range(100):
        batch = mnist.train.next_batch(100)
        ann.train_step(input_tensor_value=batch[0], output_desired=batch[1].astype(np.float32), global_iteration=it)


if __name__ == '__main__':
    main()
