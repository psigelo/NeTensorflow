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
        self.lost_function = None

    def create_loss_function(self):
        self.last_layer = self.layers_structures[-1].layers[-1]
        self.output_last_layer = self.last_layer.get_tensor()
        with tf.name_scope('DefaultTrainerDesiredOutput'):
            self.desired_output = tf.placeholder(tf.float32, [None, self.last_layer.get_input_amount()])

        self.lost_function = tf.nn.softmax_cross_entropy_with_logits(labels=self.desired_output,
                                                                     logits=self.output_last_layer)
        self.__train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.lost_function)

    @property
    def train_step(self):  # without a setter for security reasons
        return self.__train_step
