import tensorflow as tf
from keras import backend as K
from architectures import squash
from architectures.three_capsule_layers_architectures import wrappers


def vec_transformationByConv(poses, input_capsule_dim, input_capsule_num, output_capsule_dim, output_capsule_num):
    kernel = wrappers._get_weights_wrapper(
      name='weights',
      shape=[1, input_capsule_dim, output_capsule_dim * output_capsule_num],
      weights_decay_factor=0.0
    )

    u_hat_vecs = K.conv1d(poses, kernel)
    u_hat_vecs = K.reshape(u_hat_vecs, (-1, input_capsule_num, output_capsule_num, output_capsule_dim))
    u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

    return u_hat_vecs


def capsules_primary(inputs, shape, strides, padding, pose_shape, add_bias, name):
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


def capsule_fully_connected(nets, output_capsule_num, iterations, name):
    with tf.variable_scope(name):
        poses, i_activations = nets
        input_pose_shape = poses.get_shape().as_list()

        u_hat_vecs = vec_transformationByConv(
            poses,
            input_pose_shape[-1],
            input_pose_shape[1],
            input_pose_shape[-1],
            output_capsule_num,
        )

        tf.logging.info('votes shape: {}'.format(u_hat_vecs.get_shape()))

        beta_a = wrappers._get_weights_wrapper(name='beta_a', shape=[1, output_capsule_num])

        poses, activations = routing(u_hat_vecs, beta_a, iterations, output_capsule_num, i_activations)

        tf.logging.info('capsule fc shape: {}'.format(poses.get_shape()))

    return poses, activations


def capsule_convolution(nets, shape, strides, iterations, name):
    with tf.variable_scope(name):
        poses, i_activations = nets

        inputs_poses_shape = poses.get_shape().as_list()

        # vertical
        end_height_kernel = inputs_poses_shape[1] + 1 - shape[0]

        hk_offsets = [
            [(h_offset + k_offset) for k_offset in range(0, shape[0])] for h_offset in range(0, end_height_kernel, strides[1])
        ]

        # horizontal
        end_width_kernel = inputs_poses_shape[2] + 1 - shape[1]

        wk_offsets = [
            [(w_offset + k_offset) for k_offset in range(0, shape[1])] for w_offset in range(0, end_width_kernel, strides[2])
        ]

        # Poses
        poses_height_kernel = tf.gather(poses, hk_offsets, axis=1, name='gather_poses_height_kernel')
        poses_width_kernel = tf.gather(poses_height_kernel,  wk_offsets, axis=3, name='gather_poses_width_kernel')
        inputs_poses_patches = tf.transpose(poses_width_kernel, perm=[0, 1, 3, 2, 4, 5, 6], name='inputs_poses_patches')

        tf.logging.info('i_poses_patches shape: {}'.format(inputs_poses_patches.get_shape()))

        dim = shape[0] * shape[1] * shape[2]

        inputs_poses_shape = inputs_poses_patches.get_shape().as_list()
        inputs_poses_patches = tf.reshape(inputs_poses_patches, [-1, dim, inputs_poses_shape[-1]])

        # Activations
        activations_height_kernel = tf.gather(i_activations, hk_offsets, axis=1, name='activations_height_kernel')
        activations_width_kernel = tf.gather(activations_height_kernel, wk_offsets, axis=3, name='activs_width_kernel')
        i_activations_patches = tf.transpose( activations_width_kernel, perm=[0, 1, 3, 2, 4, 5], name='activ_patches')

        tf.logging.info('i_activations_patches shape: {}'.format(i_activations_patches.get_shape()))

        i_activations_patches = tf.reshape(i_activations_patches, [-1, dim])

        u_hat_vecs = vec_transformationByConv(
            inputs_poses_patches,
            inputs_poses_shape[-1],
            dim,
            inputs_poses_shape[-1],
            shape[3],
        )

        tf.logging.info('capsule conv votes shape: {}'.format(u_hat_vecs.get_shape()))

        beta_a = wrappers._get_weights_wrapper(name='beta_a', shape=[1, shape[3]])

        poses, activations = routing(u_hat_vecs, beta_a, iterations, shape[3], i_activations_patches)

        poses = tf.reshape(
            poses,
            [inputs_poses_shape[0], inputs_poses_shape[1], inputs_poses_shape[2], shape[3], inputs_poses_shape[-1]]
        )

        activations = tf.reshape(
            activations,
            [inputs_poses_shape[0], inputs_poses_shape[1], inputs_poses_shape[2], shape[3]]
        )

        nets = poses, activations

    tf.logging.info("capsule conv poses dimension:{}".format(poses.get_shape()))
    tf.logging.info("capsule conv activations dimension:{}".format(activations.get_shape()))
    return nets


def routing(u_hat_vecs, beta_a, iterations, output_capsule_num, i_activations):
    b = K.zeros_like(u_hat_vecs[:, :, :, 0])

    if i_activations is not None:
        i_activations = i_activations[..., tf.newaxis]

    for i in range(iterations):

        c = softmax(b, 1)
#       if i_activations is not None:
#           tf.transpose(tf.transpose(c, perm=[0,2,1]) * i_activations, perm=[0,2,1])

        s = K.batch_dot(c, u_hat_vecs, [2, 2])
        outputs = squash.squash_v1(s)

        if i < iterations - 1:
            b = b + K.batch_dot(outputs, u_hat_vecs, [2, 3])

    poses = outputs
    activations = K.sqrt(K.sum(K.square(poses), 2))
    return poses, activations


def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)


def capsule_flatten(nets):
    poses, activations = nets
    input_pose_shape = poses.get_shape().as_list()

    dim = input_pose_shape[1] * input_pose_shape[2] * input_pose_shape[3]

    poses = tf.reshape(poses, [-1, dim, input_pose_shape[-1]])
    activations = tf.reshape(activations, [-1, dim])

    tf.logging.info("flatten poses dimension:{}".format(poses.get_shape()))
    tf.logging.info("flatten activations dimension:{}".format(activations.get_shape()))

    return poses, activations