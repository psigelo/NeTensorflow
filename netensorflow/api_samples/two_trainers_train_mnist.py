import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

from netensorflow.ann.ANN import ANN
from netensorflow.ann.macro_layer.MacroLayer import MacroLayer
from netensorflow.ann.macro_layer.layer_structure.InputLayerStructure import InputLayerStructure
from netensorflow.ann.macro_layer.layer_structure.LayerStructure import LayerStructure, LayerType
from netensorflow.ann.macro_layer.layer_structure.layers.ConvolutionalLayerWithPoolMax2x2 import\
    ConvolutionalLayerWithPoolMax2x2
from netensorflow.ann.macro_layer.layer_structure.layers.FullConnected import FullConnected
from netensorflow.ann.macro_layer.layer_structure.layers.FullConnectedWithSoftmaxLayer import\
    FullConnectedWithSoftmaxLayer
from netensorflow.ann.macro_layer.trainers.DefaultTrainer import DefaultTrainer

'''
    Two different trainers used to train a mnist problem.
'''


def main():
    # Input / Output
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Tensorflow
    tf_sess = tf.Session()

    # Layers
    input_dim = [None, 28, 28, 1]
    dataset_dimension = [None, 784]
    convolutional_pool_layer_1 = ConvolutionalLayerWithPoolMax2x2(height_patch=5, width_patch=5, filters_amount=32,
                                                                  strides=[1, 1, 1, 1])
    convolutional_pool_layer_2 = ConvolutionalLayerWithPoolMax2x2(height_patch=5, width_patch=5, filters_amount=32,
                                                                  strides=[1, 1, 1, 1])
    logic_layer = FullConnected(inputs_amount=300)
    out_layer = FullConnectedWithSoftmaxLayer(inputs_amount=10)

    # Layer Structures
    input_layer_structure = InputLayerStructure(input_dim, dataset_dimension)
    features_layer_structure = LayerStructure('Features', layer_type=LayerType.IMAGE,
                                              layers=[convolutional_pool_layer_1, convolutional_pool_layer_2])
    logic_layer_structure = LayerStructure('Logic', layer_type=LayerType.ONE_DIMENSION, layers=[logic_layer])
    output_layer_structure = LayerStructure('Output', layer_type=LayerType.ONE_DIMENSION, layers=[out_layer])

    # Macro Layer
    macro_layers = MacroLayer(layers_structure=[input_layer_structure, features_layer_structure,
                                                logic_layer_structure, output_layer_structure])

    # Trainers
    trainer_1 = DefaultTrainer(layers_structures=[input_layer_structure, features_layer_structure,
                                                  logic_layer_structure, output_layer_structure], name='t1')
    trainer_2 = DefaultTrainer(layers_structures=[output_layer_structure], name='t2')

    # ANN
    ann = ANN(macro_layers=macro_layers, tf_session=tf_sess, base_folder='./tensorboard_logs/',
              trainer_list=[trainer_1, trainer_2])
    ann.connect_and_initialize()
    global_it = 0
    for it in range(500):
        batch = mnist.train.next_batch(200)
        ann.train_step(input_tensor_value=batch[0], output_desired=batch[1].astype(np.float32),
                       global_iteration=global_it, trainers=[trainer_2])
        print("Train iteration: ", it)
        global_it += 1

    for it in range(500):
        batch = mnist.train.next_batch(200)
        ann.train_step(input_tensor_value=batch[0], output_desired=batch[1].astype(np.float32),
                       global_iteration=global_it, trainers=[trainer_1])
        print("Train iteration: ", it)
        global_it += 1

    ann.save(save_model=True)  # save a SavedModel of tensorflow.


if __name__ == '__main__':
    main()
