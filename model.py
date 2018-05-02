import tensorflow as tf
import numpy as np
import scipy.misc


def SaveImages(images, save_path='img/sample.jpg'):
    '''
    Args: 
        images: of size [batch_size, width, height, depth].
        save_path: string name of save path.
    '''
    h, w, d = images.shape[1], images.shape[2], images.shape[3]
    square_size = 10
    img = np.zeros((h * square_size, w * square_size, 3))

    for i in range(square_size):
        for j in range(square_size):
            img[j * h:j * h + h, i * w:i * w + w, :] = images[
                i * square_size + j, :, :, :]

    scipy.misc.imsave(save_path, img)


def make_encoder(x, z_size=10, is_training=True):
    with tf.variable_scope('encoder'):
        with tf.contrib.framework.arg_scope(
            [
                tf.contrib.layers.fully_connected,
            ],
                trainable=is_training,
                activation_fn=tf.nn.relu):

            net = tf.contrib.layers.fully_connected(
                x, num_outputs=400, scope='e1')

            logits = tf.contrib.layers.fully_connected(
                net, num_outputs=z_size * 2, activation_fn=None, scope='e2')

            return tf.distributions.Normal(
                loc=logits[..., :z_size],
                scale=tf.nn.softplus(logits[..., z_size:]))


def make_decoder(z, x_shape=(28, 28, 1), is_training=True):
    x_size = np.asscalar(np.prod(x_shape))
    with tf.variable_scope('decoder'):
        with tf.contrib.framework.arg_scope(
            [
                tf.contrib.layers.fully_connected,
                tf.contrib.layers.convolution2d_transpose
            ],
                trainable=is_training,
                activation_fn=tf.nn.relu):

            net = tf.contrib.layers.fully_connected(
                z, num_outputs=400, scope='d1')

        logits = tf.contrib.layers.fully_connected(
            net, num_outputs=x_size, activation_fn=None, scope='d2')

        return logits


def make_prior(z_size=8, dtype=tf.float32):
    return tf.distributions.Normal(
        loc=tf.zeros(z_size, dtype), scale=tf.ones(z_size, dtype))
