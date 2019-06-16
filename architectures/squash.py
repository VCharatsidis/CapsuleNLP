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