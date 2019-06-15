from keras.engine import Layer
from keras.layers import Activation
from keras.layers import K
import tensorflow as tf


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


def squash_v1(s, axis=-1):
    length_s = tf.reduce_sum(s ** 2.0, axis=axis, keepdims=True) ** 0.5
    v = s * length_s / (1.0 + length_s ** 2.0)
    return v


def squash_v3(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


def squash_v4(s, axis=-1, epsilon=1e-7, name=None):
    s_squared_norm = K.sum(K.square(s), axis, keepdims=True) + K.epsilon()
    safe_norm = K.sqrt(s_squared_norm)
    scale = 1 - tf.exp(-safe_norm)
    return scale * s / safe_norm


class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        self.squash = squash

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def routing(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = K.softmax(b)
            s = K.batch_dot(c, u_hat_vecs, [2, 2])
            v = self.squash(s)

            if i < self.routings - 1:
                b = b + K.batch_dot(v, u_hat_vecs, [2, 3])

        poses = v
        activations = K.sqrt(K.sum(K.square(poses), 2))
        return poses, activations

    def call(self, u_vecs):
        poses, _ = self.routing(u_vecs)
        return poses

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
