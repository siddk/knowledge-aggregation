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

        # Create Fields for Current Base Image, Observation, and Label
        self.base, self.obs, self.label, self.counter = None, None, None, 1

    def reset(self):
        """
        Sample new base image, label from data, apply transformation to generate new observation.
        """
        idx = np.random.choice(len(self.X))
        self.base, self.label = self.X[idx], self.Y[idx]

        # Apply transformation to base and generate new observation
        self.obs = self.transform(np.copy(self.base))

        # Set Counter
        self.counter = 1

        return self.obs

    def step(self, action, probabilities):
        """
        Step with current environment, applying action (transform base again if 0), or returning final reward (if 1).
        """
        if (self.counter + 1 > self.max_len) or (action == 1):
            # Done => Compute Final Reward
            reward = self.reward(probabilities)
            return self.obs, reward, True
        else:
            # Update Obs, Increment Counter
            self.obs = self.transform(np.copy(self.base))
            self.counter += 1
            return self.obs, 0.0, False

    def reward(self, probabilities):
        return (self.logit_scalar * probabilities[self.label]) - (self.step_scalar * (self.counter + 1))

    @staticmethod
    def transform(base_image):
        """
        Transformation for MNIST => Identity
        """
        return base_image

