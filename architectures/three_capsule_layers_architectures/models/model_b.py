from architectures.three_capsule_layers_architectures import capsule_layers
from architectures.three_capsule_layers_architectures import wrappers
import tensorflow as tf
import keras.backend as K


def get_model(X, num_classes):
    poses_list = []
    for _, ngram in enumerate([3, 4, 5]):
        with tf.variable_scope('capsule_' + str(ngram)):
            nets = wrappers._conv2d_wrapper(
                X,
                shape=[ngram, 300, 1, 32],
                strides=[1, 2, 1, 1],
                padding='VALID',
                add_bias=True,
                activation_fn=tf.nn.relu,
                name='conv1'
            )

            tf.logging.info('output shape: {}'.format(nets.get_shape()))

            nets = capsule_layers.capsules_primary(
                nets,
                shape=[1, 1, 32, 16],
                strides=[1, 1, 1, 1],
                padding='VALID',
                pose_shape=16,
                add_bias=True,
                name='primary'
            )

            nets = capsule_layers.capsule_convolution(
                nets,
                shape=[3, 1, 16, 16],
                strides=[1, 1, 1, 1],
                iterations=3,
                name='conv2'
            )

            nets = capsule_layers.capsule_flatten(nets)
            poses, activations = capsule_layers.capsule_fully_connected(nets, num_classes, 3, 'fc2')
            poses_list.append(poses)

    poses = tf.reduce_mean(tf.convert_to_tensor(poses_list), axis=0)
    activations = K.sqrt(K.sum(K.square(poses), 2))

    return poses, activations