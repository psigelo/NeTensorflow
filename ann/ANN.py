import datetime

import os
import tensorflow as tf


class ANN(object):
    def __init__(self, macro_layers=None, tf_session=None, base_folder='.'):
        self.macro_layers = macro_layers
        self.tf_session = tf_session
        self.last_layer = None
        self.first_layer = None
        self.train_writer = None
        self.tf_summaries = None
        self.base_folder = base_folder

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
        self.train_writer = \
            tf.summary.FileWriter(os.path.join(time_stamp, self.base_folder, './train'), self.tf_session.graph)
        self.tf_summaries = tf.summary.merge_all()

    def initialize(self):
        self.tf_session.run(tf.global_variables_initializer())

    def run(self):
        input_tensor = self.first_layer.get_input_tensor()
        output_tensor = self.last_layer.get_tensor()
        import numpy as np
        return self.tf_session.run(output_tensor, {input_tensor: [np.random.uniform(0.0, 10.0, 3)]})

    def write_graph_and_summaries(self, global_iteration=None):
        summaries = self.tf_session.run(self.tf_summaries)
        self.train_writer.add_summary(summaries, global_iteration)
