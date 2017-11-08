import tensorflow as tf


class DefaultTrainer(object):
    def __init__(self, layers_structures=None):
        self.__loss_function = None
        self.__optimizer = None
        self.__train_step = None
        self.last_layer = None
        if not isinstance(layers_structures, list):
            raise Exception('layer_structures must be a list of layer structures')
        self.layers_structures = layers_structures
        self.optimizer_step = None
        self.output_last_layer = None
        self.desired_output = None
        self.loss_function = None

    def create_loss_function(self):
        self.last_layer = self.layers_structures[-1].layers[-1]
        self.output_last_layer = self.last_layer.get_tensor()
        with tf.name_scope('desired_output'):
            self.desired_output = tf.placeholder(tf.float32, [None, self.last_layer.get_input_amount()])

        with tf.name_scope('loss_func'):
            self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.desired_output,
                                                                                        logits=self.output_last_layer))
        tf.summary.scalar('loss', self.loss_function)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.output_last_layer, 1), tf.argmax(self.desired_output, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        self.__train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss_function)

    @property
    def train_step(self):  # without a setter for security reasons
        return self.__train_step
