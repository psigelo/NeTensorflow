import argparse

import datetime
import tensorflow as tf
import numpy as np
import time

from netensorflow.ann.ANN import ANN
from netensorflow.ann.macro_layer.MacroLayer import MacroLayer
from netensorflow.ann.macro_layer.layer_structure.InputLayerStructure import InputLayerStructure
from netensorflow.ann.macro_layer.layer_structure.LayerStructure import LayerStructure, LayerType
from netensorflow.ann.macro_layer.layer_structure.layers.ConvolutionalLayerWithPoolMax2x2 import \
    ConvolutionalLayerWithPoolMax2x2
from netensorflow.ann.macro_layer.layer_structure.layers.FullConnected import FullConnected
from netensorflow.ann.macro_layer.layer_structure.layers.FullConnectedWithSoftmaxLayer import \
    FullConnectedWithSoftmaxLayer
from netensorflow.ann.macro_layer.trainers.DefaultTrainer import DefaultTrainer


def unpack_dataset(example):
    feature = {'image_raw': tf.FixedLenFeature([], tf.string), 'classification': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example, features=feature)
    img_raw = tf.decode_raw(features['image_raw'], tf.uint8)
    img_raw_ = tf.reshape(img_raw, [7, 100, 3])
    return img_raw_, features['classification']


class MdcDataset(object):
    def __init__(self, tf_sess,  filenames_tfrecords, batch_sizes):
        self.next_elements = list()
        for it in range(len(filenames_tfrecords)):
            dataset = tf.data.TFRecordDataset(filenames_tfrecords[it])
            dataset = dataset.map(unpack_dataset)
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(batch_sizes[it])
            dataset = dataset.repeat()
            iterator = dataset.make_initializable_iterator()
            self.sess = tf_sess
            self.sess.run(iterator.initializer)
            self.next_elements.append(iterator.get_next())

    def get_next_bash(self):
        images_ = None
        classifications_ = None
        for it in range(len(self.next_elements)):
            img, classification = self.sess.run(self.next_elements[it])
            if images_ is None:
                images_ = img.reshape(-1, 7, 100, 3)
            else:
                images_ = np.concatenate([images_, img.reshape(-1, 7, 100, 3)])

            if classifications_ is None:
                classifications_ = np.array(np.eye(3)[classification -1]).reshape(-1,3)
            else:
                classifications_ = np.concatenate([classifications_,
                                                   np.array(np.eye(3)[classification - 1]).reshape(-1, 3)])

        index = np.arange(images_.shape[0])
        np.random.shuffle(index)

        return images_[index], classifications_[index]

def main(filenames_tfrecords):
    # Tensorflow
    tf_sess = tf.Session()

    mdc_dataset = MdcDataset(tf_sess, filenames_tfrecords, batch_sizes=[10, 10, 10])

    # Layers
    input_dim = [None, 7, 100, 3]
    convolutional_pool_layer_1 = ConvolutionalLayerWithPoolMax2x2(height_patch=5, width_patch=5, filters_amount=32,
                                                                  strides=[1, 1, 1, 1])
    convolutional_pool_layer_2 = ConvolutionalLayerWithPoolMax2x2(height_patch=3, width_patch=3, filters_amount=64,
                                                                  strides=[1, 1, 1, 1])
    logic_layer = FullConnected(inputs_amount=300)
    out_layer = FullConnectedWithSoftmaxLayer(inputs_amount=3)

    # Layer Structures
    input_layer_structure = InputLayerStructure(input_dim, layer_type=LayerType.IMAGE)
    features_layer_structure = LayerStructure('Features', layer_type=LayerType.IMAGE,
                                              layers=[convolutional_pool_layer_1, convolutional_pool_layer_2])
    logic_layer_structure = LayerStructure('Logic', layer_type=LayerType.ONE_DIMENSION, layers=[logic_layer])
    output_layer_structure = LayerStructure('Output', layer_type=LayerType.ONE_DIMENSION, layers=[out_layer])

    # Macro Layer
    macro_layers = MacroLayer(layers_structure=[input_layer_structure, features_layer_structure,
                                                logic_layer_structure, output_layer_structure])

    # Train
    trainer = DefaultTrainer(layers_structures=[input_layer_structure, features_layer_structure,
                                                logic_layer_structure, output_layer_structure], learning_rate=1e-4)

    # ANN
    ann = ANN(macro_layers=macro_layers, tf_session=tf_sess, base_folder='./tensorboard_logs/', trainer_list=[trainer],
              tfdbg=False)
    ann.connect_and_initialize()

    for it in range(1000):
        images, classifications = mdc_dataset.get_next_bash()
        ann.train_step(input_tensor_value=images.astype(np.float32), output_desired=classifications.astype(np.float32),
                       global_iteration=it)
        print("Train iteration: ", it)

    last_time = time.time()
    for it in range(100):
        images, classifications = mdc_dataset.get_next_bash()
    delta_t = datetime.datetime.fromtimestamp(time.time() - last_time).strftime('%S.%f')[:-3]
    print('Time to run 100 samples: ', delta_t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filenames_tfrecords", help="the path to every tfrecords to use", nargs="+")
    args = parser.parse_args()
    main(args.filenames_tfrecords)
