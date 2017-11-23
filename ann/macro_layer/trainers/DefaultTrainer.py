import tensorflow as tf
import uuid


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
    def train_step(self):  # without a setter for security reasons
        return self.__train_step
