import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import model

tf.app.flags.DEFINE_string('ckpt_dir', '/tmp/vae1', 'Runtime directory')
tf.app.flags.DEFINE_string('data_dir', '/tmp/data1', 'Data directory')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
tf.app.flags.DEFINE_integer('max_step', 10000, 'Max step.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    batch_size = 128
    z_size = 20
    with tf.Graph().as_default() as g:

        image_placeholder = tf.placeholder(
            tf.float32, shape=[batch_size, 28 * 28], name='input')

        mvn_dist = model.make_encoder(image_placeholder, z_size)
        reconstructed_logits = model.make_decoder(mvn_dist.sample())

        # Losses.
        kl_divergence_loss = tf.reduce_mean(
            mvn_dist.kl_divergence(model.make_prior(z_size)))

        # Using tf.reduce_mean will cause mode collapsing.
        reconstruction_loss = tf.reduce_sum(
            tf.losses.sigmoid_cross_entropy(
                image_placeholder,
                reconstructed_logits,
                reduction=tf.losses.Reduction.NONE))

        total_loss = kl_divergence_loss + reconstruction_loss

        global_step = tf.train.get_or_create_global_step(graph=g)
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            total_loss, global_step=global_step)

        reconstructed_images = tf.reshape(
            tf.nn.sigmoid(reconstructed_logits), [-1, 28, 28, 1])

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.ckpt_dir,
                hooks=[
                    tf.train.StopAtStepHook(FLAGS.max_step),
                    tf.train.StepCounterHook()
                ],
                save_checkpoint_secs=60) as sess:
            step = 0
            while not sess.should_stop():
                images, labels = mnist.train.next_batch(batch_size)
                step += 1
                _, loss, reconstructed_images_value = sess.run(
                    [train_op, total_loss, reconstructed_images],
                    feed_dict={image_placeholder: images})
                print('%4d, total_loss: %1.3f' % (step, loss))

                if step % 500 == 0:
                    model.SaveImages(reconstructed_images_value,
                                     'img/sample_%d.jpg' % step)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()