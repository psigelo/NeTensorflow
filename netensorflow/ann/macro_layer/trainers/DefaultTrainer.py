import os

import json
import tensorflow as tf
import uuid

from netensorflow.ann.ANNGlobals import register_netensorflow_class


@register_netensorflow_class
class DefaultTrainer(object):
    def __init__(self, layers_structures=None, name='DefaultTrainer', restore=False, learning_rate=0.9):
        # ToDO: Name algorithm to take in account problems like two trainers with same name that can not
        #       restore properly.
        self.save_and_restore_dictionary = dict()
        self.__trainer_name = None
        self.__loss_function = None
        self.__train_step = None
        self.__desired_output = None
        self.__loss_function = None
        self.__train_summary = None
        self.__accuracy = None
        self.__learning_rate = None
        self.layers_structures = None
        if not restore:
            self.trainer_name = name + '_uuid_' + uuid.uuid4().hex
            self.layers_structures = layers_structures
            self.learning_rate = learning_rate

    def create_loss_function(self):
        last_layer = self.layers_structures[-1].layers[-1]
        output_last_layer = last_layer.get_tensor()
        with tf.name_scope(self.trainer_name):
            with tf.name_scope('desired_output'):
                with tf.device('/cpu:0'):
                    self.desired_output = tf.placeholder(tf.float32, [None, last_layer.inputs_amount])

            with tf.name_scope('loss_func'):
                self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.desired_output, logits=output_last_layer))

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(output_last_layer, 1),
                                              tf.argmax(self.desired_output, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                self.train_summary = tf.summary.merge([tf.summary.scalar('loss', self.loss_function),
                                                       tf.summary.scalar('accuracy', self.accuracy)])
        var_list = list()
        for layers_structure in self.layers_structures:
            for layer in layers_structure.layers:
                var_list += layer.layer_variables
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_function, var_list=var_list)

    def trainers_hidden_placeholder_feed(self):
        feed_dict = dict()
        for layers_structure in self.layers_structures:
            for layer in layers_structure.layers:
                if layer.layer_hidden_placeholder is not None:
                    hidden_placeholder_dict = layer.layer_hidden_placeholder
                    if hidden_placeholder_dict is not None:
                        feed_dict.update({hidden_placeholder_dict['placeholder']: hidden_placeholder_dict['training']})
        return feed_dict

    def save_netensorflow_model(self, path):
        trainer_path = os.path.join(path, self.trainer_name)
        with open(trainer_path + '_internal_data.json', 'w') as fp:
            json.dump(self.save_and_restore_dictionary, fp)

    @classmethod
    def restore_netensorflow_model(cls, path, name):
        layer_path = os.path.join(path, name)
        with open(layer_path + '_internal_data.json', 'r') as fp:
            restore_json_dict = json.load(fp)

        trainer = cls(restore=True)
        for var_name in restore_json_dict:
            setattr(trainer, var_name, restore_json_dict[var_name])
        return trainer

    @property
    def train_step(self):
        return self.__train_step

    @train_step.setter
    def train_step(self, train_step):
        if isinstance(train_step, str):
            train_step = tf.get_default_graph().get_operation_by_name(train_step)
        self.__train_step = train_step
        self.save_and_restore_dictionary['train_step'] = self.__train_step.name

    @property
    def loss_function(self):
        return self.__loss_function

    @loss_function.setter
    def loss_function(self, loss_function):
        if isinstance(loss_function, str):
            loss_function = tf.get_default_graph().get_tensor_by_name(loss_function)
        self.__loss_function = loss_function
        self.save_and_restore_dictionary['loss_function'] = self.__loss_function.name

    @property
    def train_summary(self):
        return self.__train_summary

    @train_summary.setter
    def train_summary(self, train_summary):
        if isinstance(train_summary, str):
            train_summary = tf.get_default_graph().get_tensor_by_name(train_summary)
        self.__train_summary = train_summary
        self.save_and_restore_dictionary['train_summary'] = self.__train_summary.name

    @property
    def accuracy(self):
        return self.__accuracy

    @accuracy.setter
    def accuracy(self, accuracy):
        if isinstance(accuracy, str):
            accuracy = tf.get_default_graph().get_tensor_by_name(accuracy)
        self.__accuracy = accuracy
        self.save_and_restore_dictionary['accuracy'] = self.__accuracy.name

    @property
    def desired_output(self):
        return self.__desired_output

    @desired_output.setter
    def desired_output(self, desired_output):
        if isinstance(desired_output, str):
            desired_output = tf.get_default_graph().get_tensor_by_name(desired_output)
        self.__desired_output = desired_output
        self.save_and_restore_dictionary['desired_output'] = self.__desired_output.name

    @property
    def trainer_name(self):
        return self.__trainer_name

    @trainer_name.setter
    def trainer_name(self, trainer_name):
        self.__trainer_name = trainer_name
        self.save_and_restore_dictionary['trainer_name'] = self.__trainer_name

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate
        self.save_and_restore_dictionary['learning_rate'] = self.__learning_rate
