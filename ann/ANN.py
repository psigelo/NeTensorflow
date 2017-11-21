import datetime

import os

import random
import tensorflow as tf

from ann.macro_layer.layer_structure.LayerStructure import LayerType
from ann.macro_layer.layer_structure.layers.TranslatorLayerImage2OneDimension import TranslatorLayerImage2OneDimesion


class ANN(object):
    def __init__(self, macro_layers=None, tf_session=None, base_folder='.', trainer_list=[]):
        self.macro_layers = macro_layers
        self.tf_session = tf_session
        self.last_layer = None
        self.first_layer = None
        self.train_writer = None
        self.run_writer = None
        self.tf_summaries_ann = None
        self.base_folder = base_folder
        self.trainer_list = trainer_list

    def connect_and_initialize(self):
        self.connect()
        self.initialize()

    def connect(self):
        layers_refs = list()

        # first solve union problems in all layers
        prev_layer = None
        for layer_structures in self.macro_layers.layers_structure_list:
            for layer in layer_structures.layers:
                if prev_layer is not None:
                    if layer.layer_type != prev_layer.layer_type:
                        if (prev_layer.layer_type == LayerType.IMAGE) \
                                and (layer.layer_type == LayerType.ONE_DIMENSION):
                            layer_structures.add_layer_at_bottom(TranslatorLayerImage2OneDimesion())
                        else:
                            raise Exception('CaseNotDefined')
                prev_layer = layer

        # first get all layers
        for layer_structures in self.macro_layers.layers_structure_list:
            for layers in layer_structures.layers:
                layers_refs.append(layers)

        # Connect all layers
        for it in range(1, len(layers_refs)):  # Starting from second layer
            layers_refs[it].connect_layer(layers_refs[it - 1], layers_refs[it - 1].get_tensor())

        self.last_layer = layers_refs[-1]
        self.first_layer = layers_refs[0]
        time_stamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
        print("TimeStamp used: ", time_stamp)

        for trainer in self.trainer_list:  # Must do it before writers
            trainer.create_loss_function()

        self.train_writer = \
            tf.summary.FileWriter(os.path.join(self.base_folder, time_stamp,  'train'), self.tf_session.graph)
        self.run_writer = \
            tf.summary.FileWriter(os.path.join(self.base_folder, time_stamp, 'run'), self.tf_session.graph)

        summaries = list()
        for layers_structures in self.macro_layers.layers_structure_list:
                for layer in layers_structures.layers:
                    summaries += layer.summaries
        self.tf_summaries_ann = tf.summary.merge(summaries)

    def initialize(self):
        self.tf_session.run(tf.global_variables_initializer())

    def run(self, global_iteration, input_tensor_value, write_summaries=True):
        input_tensor = self.first_layer.get_input_tensor()
        output_tensor = self.last_layer.get_tensor()
        run_options = None
        run_metadata = None
        if write_summaries:
            if random.random() < 0.01:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            summaries_ann, result = self.tf_session.run([self.tf_summaries_ann, output_tensor],
                                                        feed_dict={input_tensor: input_tensor_value},
                                                        options=run_options, run_metadata=run_metadata)
            if run_metadata is not None:
                self.run_writer.add_run_metadata(run_metadata, 'step%d' % global_iteration)

            self.run_writer.add_summary(summaries_ann, global_iteration)

        else:
            result = self.tf_session.run(output_tensor, feed_dict={input_tensor: input_tensor_value})

        return result

    def train_step(self,input_tensor_value, output_desired, global_iteration,  write_summaries=True):
        for trainer in self.trainer_list:
            input_tensor = self.first_layer.get_input_tensor()
            desired_output = trainer.desired_output
            if write_summaries:
                run_options = None
                run_metadata = None
                if random.random() < 0.01:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                summaries_ann, summaries_train, _ = self.tf_session.run([self.tf_summaries_ann, trainer.train_summary,
                                                                         trainer.train_step],
                                                                        feed_dict={input_tensor: input_tensor_value,
                                                                                   desired_output: output_desired},
                                                                        options=run_options, run_metadata=run_metadata)
                if run_metadata is not None:
                    self.train_writer.add_run_metadata(run_metadata, 'step%d' % global_iteration)
                self.train_writer.add_summary(summaries_ann, global_iteration)
                self.train_writer.add_summary(summaries_train, global_iteration)
            else:
                self.tf_session.run( trainer.train_step, feed_dict={input_tensor: input_tensor_value,
                                                                    desired_output: output_desired})
