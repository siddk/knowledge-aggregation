"""
noisy_mnist.py

Distributed Knowledge Aggregation for Noisy MNIST Experiment.

Sampling Procedure:
    - Pick MNIST Digit
    - Pick amount of Noise - Uniform from [50, 700] => amount_noise => Indices in MNIST Image to Corrupt
    - Sample amount_noise times from Standard Normal => Replace Corruption Indices with samples
"""
from environments.noisy_mnist import NoisyMnist
from models.mnist import MnistKAN
from keras.datasets import mnist
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("n_threads", 32, "Number of environments to run in parallel.")


def main(_):
    # Load Datasets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Instantiate Model
    mnist_kan = MnistKAN()

    # Create environments
    envs = [NoisyMnist(x_train, y_train) for _ in range(FLAGS.n_threads)]

    # Fit Model
    running_reward = None
    for e in range(2000):
        episode_rs = mnist_kan.fit(envs, e)
        for er in episode_rs:
            running_reward = er if running_reward is None else (0.99 * running_reward + 0.01 * er)

        if e % 10 == 0:
            print 'Batch %d (episode %d), batch avg. reward: %.2f, running reward: %.3f' % (e, (e + 1) * FLAGS.n_threads, np.mean(episode_rs), running_reward)


if __name__ == "__main__":
    tf.app.run()