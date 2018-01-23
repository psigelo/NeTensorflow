import datetime
import os
import random
import shutil
import uuid
import json

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from netensorflow.ann.ANNGlobals import NETENSORFLOW_CLASSES
from netensorflow.ann.macro_layer import MacroLayer
from netensorflow.ann.macro_layer.layer_structure.LayerStructure import LayerType

from netensorflow.ann.macro_layer.layer_structure.layers.TranslatorLayerImage2OneDimension import   \
    TranslatorLayerImage2OneDimension
from tensorflow.python import debug as tf_debug


class ANN(object):
    def __init__(self, macro_layers=None, tf_session=None, base_folder='.', trainer_list=list(), restore=False,
                 tfdbg=False):
        self.save_and_restore_dictionary = dict()
        self.tf_session = tf_session
        self.__id = uuid.uuid4().hex
        self.__tf_summaries_ann = None
        self.__base_folder = base_folder
        self.__time_stamp = None
        self.__best_accuracy = 0.0
        self.model_is_saved = False
        self.trainer_list = None
        self.macro_layers = None
        self.train_writer = None
        self.run_writer = None
        self.saver = None
        self.last_layer = None
        self.first_layer = None
        self.tfdbg = tfdbg
        if not restore:
            self.trainer_list = trainer_list
            self.macro_layers = macro_layers
            self.model_is_saved = False
        else:
            self.model_is_saved = True  # actually is being restore from the model.

    def connect_and_initialize(self):
        self.connect()
        self.initialize()

    def connect(self):
        layers_refs = list()
        # first solve union problems in all layers (cases like layer from OneDimension to layer with more dims)
        prev_layer = None
        for layer_structures in self.macro_layers.layers_structure_list:
            for layer in layer_structures.layers:
                if prev_layer is not None:
                    if layer.layer_type != prev_layer.layer_type:
                        if (prev_layer.layer_type == LayerType.IMAGE) \
                                and (layer.layer_type == LayerType.ONE_DIMENSION):
                            layer_structures.add_layer_at_bottom(TranslatorLayerImage2OneDimension())
                        else:
                            raise Exception('CaseNotDefined')
                prev_layer = layer

        # get all layers
        for layer_structures in self.macro_layers.layers_structure_list:
            for layers in layer_structures.layers:
                layers_refs.append(layers)

        # Connect all layers
        for it in range(1, len(layers_refs)):  # Starting from second layer
            layers_refs[it].connect_layer(layers_refs[it - 1], layers_refs[it - 1].get_tensor())

        self.last_layer = layers_refs[-1]
        self.first_layer = layers_refs[0]
        self.time_stamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
        print("TimeStamp used: ", self.time_stamp)

        for trainer in self.trainer_list:  # Must do it before writers
            trainer.create_loss_function()

        self.train_writer = \
            tf.summary.FileWriter(os.path.join(self.base_folder, self.time_stamp, 'train'), self.tf_session.graph)
        self.run_writer = \
            tf.summary.FileWriter(os.path.join(self.base_folder, self.time_stamp, 'run'), self.tf_session.graph)

        summaries = list()
        for layers_structures in self.macro_layers.layers_structure_list:
                for layer in layers_structures.layers:
                    summaries += layer.summaries
        self.tf_summaries_ann = tf.summary.merge(summaries)
        self.saver = tf.train.Saver(max_to_keep=None)

    def initialize(self):
        self.tf_session.run(tf.global_variables_initializer())
        if self.tfdbg:
            self.tf_session = tf_debug.LocalCLIDebugWrapperSession(self.tf_session)

    def run(self, global_iteration, input_tensor_value, write_summaries=True):
        input_tensor = self.first_layer.get_input_tensor()
        output_tensor = self.last_layer.get_tensor()
        run_options = None
        run_metadata = None
        feed_dict = {input_tensor: input_tensor_value}
        for layers_structures in self.macro_layers.layers_structure_list:
                for layer in layers_structures.layers:
                    if layer.layer_hidden_placeholder is not None:
                        hidden_placeholder_dict = layer.layer_hidden_placeholder
                        if hidden_placeholder_dict is not None:
                            feed_dict.update(
                                {hidden_placeholder_dict['placeholder']: hidden_placeholder_dict['running']})
        if write_summaries:
            if random.random() < 0.01:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            summaries_ann, result = self.tf_session.run([self.tf_summaries_ann, output_tensor],
                                                        feed_dict=feed_dict,
                                                        options=run_options, run_metadata=run_metadata)
            if run_metadata is not None:
                self.run_writer.add_run_metadata(run_metadata, 'step%d' % global_iteration)

            self.run_writer.add_summary(summaries_ann, global_iteration)

        else:
            result = self.tf_session.run(output_tensor, feed_dict=feed_dict)

        return result

    def train_step(self, input_tensor_value, output_desired, global_iteration,  write_summaries=True, trainers=None,
                   verbose=True, run_performance_store_prob=0.1, save_best_accuracy_train=False):
        for trainer in self.trainer_list:
            if trainers is not None:
                if trainer.name not in list(map(lambda x: x.name, trainers)):
                    continue
            input_tensor = self.first_layer.get_input_tensor()
            desired_output = trainer.desired_output
            feed_dict = {input_tensor: input_tensor_value, desired_output: output_desired}
            feed_dict.update(trainer.trainers_hidden_placeholder_feed())

            if write_summaries:
                run_options = None
                run_metadata = None
                if random.random() < run_performance_store_prob:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                summ_ann, summ_train, _, accuracy = self.tf_session.run([self.tf_summaries_ann, trainer.train_summary,
                                                                         trainer.train_step, trainer.accuracy],
                                                                        feed_dict=feed_dict,
                                                                        options=run_options, run_metadata=run_metadata)
                print("train accuracy: ", accuracy)
                if save_best_accuracy_train:
                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        if verbose:
                            print("New accuracy obtained: ", self.best_accuracy)
                        if not self.model_is_saved:
                            self.model_is_saved = True
                        self.save(check_point_iteration=True, iteration=global_iteration, save_model=True)
                if run_metadata is not None:
                    self.train_writer.add_run_metadata(run_metadata, 'step%d' % global_iteration)
                self.train_writer.add_summary(summ_ann, global_iteration)
                self.train_writer.add_summary(summ_train, global_iteration)
            else:
                _, accuracy = self.tf_session.run(trainer.train_step, trainer.accuracy, feed_dict=feed_dict)
            return accuracy

    def save_best_validate_accuracy(self, trainers, input_validation_set, output_validation_set, global_iteration):
        for trainer in self.trainer_list:
            if trainers is not None:
                if trainer.name not in list(map(lambda x: x.name, trainers)):
                    continue
            input_tensor = self.first_layer.get_input_tensor()
            desired_output = trainer.desired_output
            feed_dict = {input_tensor: input_validation_set, desired_output: output_validation_set}
            feed_dict.update(trainer.trainers_hidden_placeholder_feed())
            accuracy_validation = self.tf_session.run([trainer.accuracy], feed_dict=feed_dict)
            print("accuracy_validation: ", accuracy_validation)
            if accuracy_validation > self.best_accuracy:
                self.best_accuracy = accuracy_validation
                if not self.model_is_saved:
                    print("New Champion with validation dataset")
                    self.model_is_saved = True
                self.save(check_point_iteration=True, iteration=global_iteration, save_model=True)
            return accuracy_validation

    def save(self, check_point_iteration=False, iteration=None, save_model=False):
        save_base_folder = os.path.join(os.path.join(self.base_folder, self.time_stamp), 'ANN_STORE_checkpoint')
        ann_folder = os.path.join(save_base_folder, self.id)
        if save_model:
            # Saving the tensorflow Model through the SavedModel builder
            saved_model_path = ann_folder + self.id + '_model_'
            if os.path.exists(saved_model_path):
                shutil.rmtree(saved_model_path, ignore_errors=True)
            nt_saved_model_path = ann_folder + self.id + '_netensorflow_'
            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
            builder.add_meta_graph_and_variables(self.tf_session, [tag_constants.TRAINING])
            builder.save()
            # Now saving the netensorflow ann structure, to be available to load all needed in future
            self.save_netensorflow_model(nt_saved_model_path)

        if not os.path.exists(save_base_folder):
            os.mkdir(save_base_folder)
        if not os.path.exists(ann_folder):
            os.mkdir(ann_folder)
        if check_point_iteration:
            path = os.path.join(ann_folder, self.id + '_it_' + str(iteration) + '.ckpt')
            print("Saved path: ", path)
            self.saver.save(self.tf_session, path)
        else:
            path = os.path.join(ann_folder, self.id + '.ckpt')
            print("Saved path: ", path)
            self.saver.save(self.tf_session, path)

    def save_netensorflow_model(self, path):
        ann_path = os.path.join(path, 'ann')
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(ann_path):
            os.mkdir(ann_path)

        store_dict = dict()
        store_dict['layers_structures'] = [(ls.name, ls.__class__.__name__)
                                           for ls in self.macro_layers.layers_structure_list]
        store_dict['trainers'] = [(trainer.trainer_name, trainer.__class__.__name__) for trainer in self.trainer_list]

        with open(ann_path + '_data.json', 'w') as fp:
            json.dump(store_dict, fp)

        with open(ann_path + '_internal_data.json', 'w') as fp:
            json.dump(self.save_and_restore_dictionary, fp)

        for layer_structure in self.macro_layers.layers_structure_list:
            layer_structure.save_netensorflow_model(ann_path)

        for trainer in self.trainer_list:
            trainer.save_netensorflow_model(ann_path)

    @staticmethod
    def restore_netensorflow_model(path, tf_session, base_folder='.', preserve_time_stamp=False):
        ann_path = os.path.join(path + '_netensorflow_', 'ann')
        with open(ann_path + '_data.json', 'r') as fp:
            data_json = json.load(fp)

        tensorflow_saved_model_path = path + '_model_'
        tf.saved_model.loader.load(tf_session, [tag_constants.TRAINING], tensorflow_saved_model_path)

        layers_structure_list = list()
        for layer_structure_name, layer_structure_class_name in data_json['layers_structures']:
            layer_structure_class = NETENSORFLOW_CLASSES[layer_structure_class_name]
            layers_structure_list.append(layer_structure_class.restore_netensorflow_model(ann_path,
                                                                                          layer_structure_name))
        trainers = list()
        for trainers_name, trainers_class_name in data_json['trainers']:
            trainer_class = NETENSORFLOW_CLASSES[trainers_class_name]
            trainers.append(trainer_class.restore_netensorflow_model(ann_path, trainers_name))

        macro_layer = MacroLayer(layers_structure=layers_structure_list)
        ann = ANN(macro_layers=macro_layer, base_folder=base_folder, trainer_list=trainers, tf_session=tf_session)

        with open(ann_path + '_internal_data.json', 'r') as fp:
            restore_json_dict = json.load(fp)

        for var_name in restore_json_dict:
            setattr(ann, var_name, restore_json_dict[var_name])

        if not preserve_time_stamp:
            ann.time_stamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())

        ann.first_layer = layers_structure_list[0].layers[0]
        ann.last_layer = layers_structure_list[-1].layers[-1]

        ann.train_writer = \
            tf.summary.FileWriter(os.path.join(ann.base_folder, ann.time_stamp, 'train'), ann.tf_session.graph)
        ann.run_writer = \
            tf.summary.FileWriter(os.path.join(ann.base_folder, ann.time_stamp, 'run'), ann.tf_session.graph)
        ann.saver = tf.train.Saver(max_to_keep=None)

        return ann

    def get_input_dimension(self):
        return self.first_layer.inputs_dimension

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, _id):
        self.__id = _id
        self.save_and_restore_dictionary['id'] = self.__id

    @property
    def tf_summaries_ann(self):
        return self.__tf_summaries_ann

    @tf_summaries_ann.setter
    def tf_summaries_ann(self, tf_summaries_ann):
        if isinstance(tf_summaries_ann, str):
            tf_summaries_ann = tf.get_default_graph().get_tensor_by_name(tf_summaries_ann)
        self.__tf_summaries_ann = tf_summaries_ann
        self.save_and_restore_dictionary['tf_summaries_ann'] = self.__tf_summaries_ann.name

    @property
    def base_folder(self):
        return self.__base_folder

    @base_folder.setter
    def base_folder(self, base_folder):
        self.__base_folder = base_folder
        self.save_and_restore_dictionary['base_folder'] = self.__base_folder

    @property
    def time_stamp(self):
        return self.__time_stamp

    @time_stamp.setter
    def time_stamp(self, time_stamp):
        self.__time_stamp = time_stamp
        self.save_and_restore_dictionary['time_stamp'] = self.__time_stamp

    @property
    def best_accuracy(self):
        return self.__best_accuracy

    @best_accuracy.setter
    def best_accuracy(self, best_accuracy):
        if isinstance(best_accuracy, str):
            best_accuracy = float(best_accuracy)
        self.__best_accuracy = best_accuracy
        self.save_and_restore_dictionary['best_accuracy'] = str(self.__best_accuracy)
