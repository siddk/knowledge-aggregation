"""
noisy_mnist.py

Distributed Knowledge Aggregation for Noisy MNIST Experiment.

Sampling Procedure:
    - Pick MNIST Digit
    - Pick amount of Noise - Uniform from [50, 700] => amount_noise => Indices in MNIST Image to Corrupt
    - Sample amount_noise times from Standard Normal => Replace Corruption Indices with samples
"""
import tensorflow as tf


def main(_):
    pass


if __name__ == "__main__":
    tf.app.run()