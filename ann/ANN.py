import datetime

import os
import tensorflow as tf


class ANN(object):
    def __init__(self, macro_layers=None, tf_session=None, base_folder='.', trainer_list=None):
        self.macro_layers = macro_layers
        self.tf_session = tf_session
        self.last_layer = None
        self.first_layer = None
        self.train_writer = None
        self.run_writer = None
        self.tf_summaries = None
        self.base_folder = base_folder
        if not isinstance(trainer_list, list):
            raise Exception('TrainerListIsNotList')
        self.trainer_list = trainer_list

    def connect_and_initialize(self):
        self.connect()
        self.initialize()

    def connect(self):
        layers_refs = list()

        # first get all layers
        for layer_structures in self.macro_layers.layers_structure_list:
            for layers in layer_structures.layers:
                layers_refs.append(layers)

        # Connect all layers
        for it in range(1, len(layers_refs)):  # Starting from second layer
            layers_refs[it].connect_layer(layers_refs[it - 1].get_input_amount(), layers_refs[it - 1].get_tensor())

        self.last_layer = layers_refs[-1]
        self.first_layer = layers_refs[0]
        time_stamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
        print("TimeStamp used: ", time_stamp)
        self.train_writer = \
            tf.summary.FileWriter(os.path.join(self.base_folder, time_stamp,  'train'), self.tf_session.graph)
        self.run_writer = \
            tf.summary.FileWriter(os.path.join(self.base_folder, time_stamp, 'run'), self.tf_session.graph)
        self.tf_summaries = tf.summary.merge_all()
        for trainer in self.trainer_list:
            trainer.create_loss_function()

    def initialize(self):
        self.tf_session.run(tf.global_variables_initializer())

    def run(self, global_iteration, input_tensor_value, write_summaries=True):
        input_tensor = self.first_layer.get_input_tensor()
        output_tensor = self.last_layer.get_tensor()
        result = self.tf_session.run(output_tensor, feed_dict={input_tensor: input_tensor_value})
        if write_summaries:
            self.write_graph_and_summaries(global_iteration=global_iteration, writer='run')
        return result

    def train_step(self,input_tensor_value, output_desired, global_iteration,  write_summaries=True):
        for trainer in self.trainer_list:
            self.tf_session.run(trainer.train_step)
        if write_summaries:
            self.write_graph_and_summaries(global_iteration=global_iteration, writer='run')

    def write_graph_and_summaries(self, global_iteration=None, writer=None):
        if writer == 'train':
            tf_writer = self.train_writer
        elif writer == 'run':
            tf_writer = self.run_writer
        else:
            raise Exception('writer must be train or run')
        summaries = self.tf_session.run(self.tf_summaries)
        tf_writer.add_summary(summaries, global_iteration)

    def get_last_lost_functions(self):
        pass  # return something like [{name: 'Default', iteration: 9, value: 0.0004214}, {name: ...}]