import tensorflow as tf


def variable_summaries(var):
    """
        Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        a = tf.summary.scalar('stddev', stddev)
        b = tf.summary.scalar('max', tf.reduce_max(var))
        c = tf.summary.scalar('min', tf.reduce_min(var))
        d = tf.summary.histogram('histogram', var)

    return list([a, b, c, d])
