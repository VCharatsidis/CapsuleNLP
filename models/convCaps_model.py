from keras.layers import SpatialDropout1D
from keras.models import Model
from keras.layers import Bidirectional, Dense, Dropout, Embedding, Flatten
from keras.layers import GRU, Input
import keras
import tensorflow as tf
from Capsule import Capsule
from keras import backend as K
import squash
import wrappers



def vec_transformationByConv(poses, input_capsule_dim, input_capsule_num, output_capsule_dim, output_capsule_num):
    kernel = wrappers._get_weights_wrapper(
      name='weights', shape=[1, input_capsule_dim, output_capsule_dim*output_capsule_num], weights_decay_factor=0.0
    )

    u_hat_vecs = keras.backend.conv1d(poses, kernel)
    u_hat_vecs = keras.backend.reshape(u_hat_vecs, (-1, input_capsule_num, output_capsule_num, output_capsule_dim))
    u_hat_vecs = keras.backend.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

    return u_hat_vecs


def capsules_init(inputs, shape, strides, padding, pose_shape, add_bias, name):
    with tf.variable_scope(name):
        poses = wrappers._conv2d_wrapper(
          inputs,
          shape=shape[0:-1] + [shape[-1] * pose_shape],
          strides=strides,
          padding=padding,
          add_bias=add_bias,
          activation_fn=None,
          name='pose_stacked'
        )
        poses_shape = poses.get_shape().as_list()
        poses = tf.reshape(poses, [-1, poses_shape[1], poses_shape[2], shape[-1], pose_shape])

        beta_a = wrappers._get_weights_wrapper(name='beta_a', shape=[1, shape[-1]])

        poses = squash.squash_v1(poses, axis=-1)
        activations = K.sqrt(K.sum(K.square(poses), axis=-1)) + beta_a
        tf.logging.info("prim poses dimension:{}".format(poses.get_shape()))

    return poses, activations
