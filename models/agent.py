import tensorflow as tf
import numpy as np

def leaky_relu(x, alpha=0.001):
    return tf.maximum(x, alpha*x)

def encode_image(image, name, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        c1 = tf.layers.conv2d(image, 16, 5, activation=leaky_relu, strides=2, name='c1') # 14 x 14 x 16
        c2 = tf.layers.conv2d(c1, 32, 5, activation=leaky_relu, strides=2, name='c2') # 7 x 7 x 32
        fc1 = tf.layers.dense(tf.reshape(c2, [-1, 7*7*32]), 100, activation=leaky_relu, name='fc1')
    return fc1

def to_policy_vector(hidden_state, num_classes, name, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        out = tf.layers.dense(hidden_state, 2*num_classes, name='out')
    return out




class Agent(object):


    def __init__(self, max_iters=5, num_classes=10):
        self.max_iters = max_iters
        self.num_classes = num_classes
        self.sess = tf.Session()

    def build_network(self):
        self.inp_images = tf.placeholder(tf.float32, [None, self.max_iters, 28, 28, 1])
        self.inp_selected_policies = tf.placeholder(tf.int32, [None])
        self.inp_reward = tf.placeholder(tf.float32, [None])
        self.inp_advantage = tf.placeholder(tf.float32, [None])
        self.cell = tf.contrib.rnn.GRUCell(100)
        ############
        # This part of the code creates the policy vector of size [batch_size, max_iters, 2*num_classes]
        hidden_inputs = []
        for i in range(self.max_iters):
            encoding = encode_image(self.inp_images[:, i, :, :, :], 'encoder', reuse=i > 0)
            hidden_inputs.append(encoding)
        hidden_inputs = tf.transpose(hidden_inputs, [1, 0, 2]) # [batch_size, max_iters, hidden_size]
        # GRU - RNN with Empty Initial State
        rnn_outputs, _ = tf.nn.dynamic_rnn(self.cell, hidden_inputs, dtype=tf.float32,
                                                          sequence_length=self.max_iters, scope='rnn')
        all_policies = []
        for i in range(self.max_iters):
            policy = to_policy_vector(rnn_outputs[:, i, :], self.num_classes, 'policy', reuse=i > 0)
            all_policies.append(policy)
        self.all_policies = tf.transpose(all_policies, [1, 0, 2]) # [bs, max_iters, 2*num_classes]

        # This part of the code selects the policy using the first "accept" policy.
        policy_selector = tf.reshape(tf.one_hot(self.inp_selected_policies, self.max_iters), [-1, self.max_iters, 1])
        self.selected_policy = tf.reduce_sum(self.all_policies * policy_selector, axis=1) # [bs, 2*num_classes]
        self.selected_policy_choice = tf.argmax(self.selected_policy_choice, axis=1) # [bs]
        ############
        # TODO Apply actor critic reward / advantage stuff here.

    def softmax(self, p):
        e_x = np.exp(p - np.max(p))
        return e_x / e_x.sum(axis=0)

    def sample_from_policies(self, policies):
        choices = np.zeros([policies.shape[0], policies.shape[1]])
        for i in range(policies.shape[0]):
            for j in range(policies.shape[1]):
                choices[i, j] = np.random.choice(range(2*self.num_classes), p=self.softmax(policies[i, j, :]))
        return choices

    def train_step(self, images):
        # get the policy values for the image inputs.
        [policy_values] = self.sess.run([self.all_policies], feed_dict={self.inp_images: images})
        policy_choices = self.sample_from_policies(policy_values) # [bs, max_iters]
        # ensure that the last policy choice is always an "accept" policy (action between 10-20)
        policy_choices[:, -1] %= self.num_classes
        policy_choices[:, -1] += self.num_classes
        # grab the first "accept policy" for each element in the batch
        selected_policies = np.argmax(policy_choices >= self.num_classes, axis=1) # [bs]
        # TODO perform the training operation on the selected policy.
        #[] = self.sess.run([], feed_dict={self.inp_images: images,
        #                                  self.inp_selected_policies: selected_policies})






