import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from netensorflow.ann.ANN import ANN
import sys

'''
    Load an ANN from a previously saved ANN like in two_trainers_train_mnist.py.

    Usage:

    python ann_restore.py directory_netensorflow_model_saved

    where directory_netensorflow_model_saved must be the actual directory of the saved netensorflow model.
'''


def main():
    # Input / Output
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Tensorflow
    tf_sess = tf.Session()

    # ANN
    ann = ANN.restore_netensorflow_model(sys.argv[1], tf_sess, base_folder='./tensorboard_logs/')

    trainer_1 = ann.trainer_list[0]
    trainer_2 = ann.trainer_list[1]

    # Re-starting train and see if all is ok.
    global_it = 0
    for it in range(500):
        batch = mnist.train.next_batch(200)
        ann.train_step(input_tensor_value=batch[0], output_desired=batch[1].astype(np.float32),
                       global_iteration=global_it, trainers=[trainer_2])
        print("Train iteration: ", it)
        global_it += 1

    for it in range(10000):
        batch = mnist.train.next_batch(200)
        ann.train_step(input_tensor_value=batch[0], output_desired=batch[1].astype(np.float32),
                       global_iteration=global_it, trainers=[trainer_1])
        print("Train iteration: ", it)
        global_it += 1


if __name__ == '__main__':
    main()
