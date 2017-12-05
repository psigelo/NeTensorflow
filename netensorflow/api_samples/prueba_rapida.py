import argparse

import datetime
from functools import reduce

import tensorflow as tf
import numpy as np
import time

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

        return images_[index], classifications_[index].astype(np.float32)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(filenames_tfrecords):
    # Tensorflow
    tf_sess = tf.Session()

    mdc_dataset = MdcDataset(tf_sess, filenames_tfrecords, batch_sizes=[30, 40, 50])

    # Create the model
    x = tf.placeholder(tf.float32, [None, 7, 100, 3])
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 3])

    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([3200, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 2 * 25 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 3])
    b_fc2 = bias_variable([3])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf_sess.run(tf.global_variables_initializer())
    prev_best = None

    for _ in range(50000):
        images, classifications = mdc_dataset.get_next_bash()
        _, accuracy_val = tf_sess.run([train_step, accuracy], feed_dict={x: images, y_: classifications, keep_prob:0.5})
        if prev_best is None:
            prev_best = accuracy_val
            print('accuracy:', accuracy_val)
        else:
            if prev_best < accuracy_val:
                prev_best = accuracy_val
                print('accuracy:', accuracy_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filenames_tfrecords", help="the path to every tfrecords to use", nargs="+")
    args = parser.parse_args()
    main(args.filenames_tfrecords)
