from architectures.three_capsule_layers_architectures import wrappers
import tensorflow as tf
import tensorflow.contrib.slim as slim


def get_model(X, num_classes):
    nets = wrappers._conv2d_wrapper(
        X,
        shape=[3, 300, 1, 32],
        strides=[1, 1, 1, 1],
        padding='VALID',
        add_bias=False,
        activation_fn=tf.nn.relu,
        name='conv1'
    )

    nets = slim.flatten(nets)
    tf.logging.info('flatten shape: {}'.format(nets.get_shape()))

    nets = slim.fully_connected(nets, 128, scope='relu_fc3', activation_fn=tf.nn.relu)
    tf.logging.info('fc shape: {}'.format(nets.get_shape()))

    activations = tf.sigmoid(slim.fully_connected(nets, num_classes, scope='final_layer', activation_fn=None))
    tf.logging.info('fc shape: {}'.format(activations.get_shape()))

    return tf.zeros([0]), activations