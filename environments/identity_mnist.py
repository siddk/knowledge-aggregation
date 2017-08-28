"""
identity_mnist.py

Environment for Identity MNIST Experiment.
"""
import numpy as np


class IdentityMnist:
    def __init__(self, X, Y, max_len=5, logit_scalar=5.0, step_scalar=1.0, seed=None):
        """
        Initialize a Noisy MNIST Environment, with the given Training Data.
        """
        self.X, self.Y, self.max_len = np.expand_dims(X, axis=-1), Y, max_len
        self.logit_scalar, self.step_scalar = logit_scalar, step_scalar

        # Check if Seed
        if seed is not None:
            self.X, self.Y = [self.X], [self.Y]
            np.random.seed(seed)
        else:
            assert(len(self.X) == len(self.Y))

        # Create Fields for Current Base Image, Observations, and Label
        self.base, self.obs, self.label = None, None, None

    def reset(self):
        """
        Sample new base image, label from data, apply transformations to generate new observation.
        """
        idx = np.random.choice(len(self.X))
        self.base, self.label = self.X[idx], self.Y[idx]

        # Apply transformation to base and generate new observation
        self.obs = self.transform(np.copy(self.base))

        return self.obs

    def reward(self, probabilities, counter):
        return (self.logit_scalar * probabilities[self.label]) - (self.step_scalar * counter)

    def transform(self, base_image):
        """
        Transformation for MNIST => Identity
        """
        return [base_image for _ in range(self.max_len)]

