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
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("n_threads", 8, "Number of environments to run in parallel.")


def main(_):
    # Load Datasets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Instantiate Model
    mnist_kan = MnistKAN()

    # Create environments
    envs = [NoisyMnist(x_train, y_train) for _ in range(FLAGS.n_threads)]

    # Fit Model
    for _ in range(1):
        mnist_kan.fit(envs)


if __name__ == "__main__":
    tf.app.run()