import tensorflow as tf
import uuid

from netensorflow.ann.ANNGlobals import register_netensorflow_class


@register_netensorflow_class
class DefaultTrainer(object):
    def __init__(self, layers_structures=None, name='DefaultTrainer'):
        self.uuid = uuid.uuid4().hex
        self.__loss_function = None
        self.__optimizer = None
        self.__train_step = None
        self.last_layer = None
        self.name = name
        if not isinstance(layers_structures, list):
            raise Exception('layer_structures must be a list of layer structures')
        self.layers_structures = layers_structures
        self.optimizer_step = None
        self.output_last_layer = None
        self.desired_output = None
        self.loss_function = None
        self.train_summary = None
        self.accuracy = None

    def create_loss_function(self):
        self.last_layer = self.layers_structures[-1].layers[-1]
        self.output_last_layer = self.last_layer.get_tensor()
        with tf.name_scope(self.name):
            with tf.name_scope('desired_output'):
                self.desired_output = tf.placeholder(tf.float32, [None, self.last_layer.inputs_amount])

            with tf.name_scope('loss_func'):
                self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.desired_output, logits=self.output_last_layer))

            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct_prediction = tf.equal(tf.argmax(self.output_last_layer, 1),
                                                  tf.argmax(self.desired_output, 1))
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                self.train_summary = tf.summary.merge([tf.summary.scalar('loss', self.loss_function),
                                                       tf.summary.scalar('accuracy', self.accuracy)])
        var_list = list()
        for layers_str in self.layers_structures:
            for layer in layers_str.layers:
                var_list += layer.layer_variables

        self.__train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss_function, var_list=var_list)

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
