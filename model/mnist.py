"""
mnist.py

Distributed Knowledge Aggregation Network with MNIST CNN Architecture.
"""
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import numpy as np
import tensorflow as tf


class MnistKAN:
    def __init__(self, img_rows=28, img_cols=28, cnn_depths=(32, 64), cnn_filters=((3, 3), (3, 3)), pool_dim=(2, 2),
                 hidden_sz=128, rnn_sz=64, submodule_sz=64, num_classes=10, num_actions=2, critic_discount=0.5):
        """
        Initialize an MNIST Knowledge Aggregation Network, with the necessary hyperparameters.
        """
        self.img_rows, self.img_cols, self.num_classes, self.num_actions = img_rows, img_cols, num_classes, num_actions
        self.cnn_depths, self.cnn_filters, self.pool_dim = cnn_depths, cnn_filters, pool_dim
        self.rnn_sz, self.hidden_sz, self.submodule_sz = rnn_sz, hidden_sz, submodule_sz
        self.critic_discount = critic_discount
        self.session = tf.Session()

        # Setup Inference Placeholders
        self.Image = tf.placeholder(tf.float32, shape=[None, self.img_rows, self.img_cols, 1], name='Image_Sample')
        self.Action = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='Action_Taken')
        self.Label = tf.placeholder(tf.int32, shape=[None], name='True_Label')
        self.Reward = tf.placeholder(tf.float32, shape=[None, 1], name='Reward')
        self.Advantage = tf.placeholder(tf.float32, shape=[None, 1], name='Advantage')
        self.Dropout = tf.placeholder(tf.float32, name='Dropout_Probability')

        # Setup RNN State Placeholder
        self.RNN_State = tf.placeholder(tf.float32, shape=[None, self.rnn_sz], name='Initial_RNN_State')

        # Compute logits, policy, and value estimate
        self.logits, self.policy, self.value = self.forward()

        # Compute supervised loss, actor-critic loss
        self.supervised_loss, self.ac_loss = self.loss()

        # Build Separate Training Operations for Losses
        self.supervised_train_op = tf.train.AdamOptimizer().minimize(self.supervised_loss)
        self.ac_train_op = tf.train.AdamOptimizer().minimize(self.ac_loss)

        # Initialize all Variables
        self.session.run(tf.global_variables_initializer())

    def forward(self):
        # First Convolution Layer
        conv1 = Conv2D(self.cnn_depths[0], kernel_size=self.cnn_filters[0], activation='relu')(self.Image)

        # Second Convolutional Layer
        conv2 = Conv2D(self.cnn_depths[1], kernel_size=self.cnn_filters[1], activation='relu')(conv1)

        # Pooling Layer
        pool = MaxPooling2D(pool_size=self.pool_dim)(conv2)

        # Flatten + Dropout
        flat_dropout = tf.nn.dropout(Flatten()(pool), self.Dropout)

        # Dense Layer
        hidden = Dense(128, activation='relu')(flat_dropout)
        hidden = tf.expand_dims(hidden, axis=1)                                              # Shape: [None, 1, hidden]

        # GRU - RNN seeded with Input Initial State
        cell = tf.contrib.rnn.GRUCell(self.rnn_sz)
        _, self.final_state = tf.nn.dynamic_rnn(cell, hidden, initial_state=self.RNN_State)  # Shape: [None, rnn_sz]

        # Logits Hidden Layer => Linear
        l_hidden = Dense(self.submodule_sz, activation='relu')(self.final_state)
        logits = Dense(self.num_classes, activation='linear')(l_hidden)

        # Policy Hidden Layer => Softmax
        p_hidden = Dense(self.submodule_sz, activation='relu')(self.final_state)
        policy = Dense(self.num_actions, activation='softmax')(p_hidden)

        # Value Hidden Layer => Linear
        v_hidden = Dense(self.submodule_sz, activation='relu')(self.final_state)
        value = Dense(1, activation='linear')(v_hidden)

        return logits, policy, value

    def predict(self, images, dropout, rnn_states):
        return self.session.run([self.logits, self.policy, self.value, self.final_state],
                                feed_dict={self.Image: images, self.RNN_State: rnn_states, self.Dropout: dropout})

    def act(self, policies):
        return [np.random.choice(self.num_actions, p=p) for p in policies]

    def loss(self):
        # Supervised Learning Loss
        supervised_loss = tf.losses.sparse_softmax_cross_entropy(self.Label, self.logits)

        # Policy Gradient (Actor) Loss
        actor_loss = -tf.reduce_sum(tf.log(self.policy) * self.Action * self.Advantage)

        # Value Function (Critic) Loss
        critic_loss = 0.5 * tf.reduce_sum(tf.square((self.value - self.Reward)))

        return supervised_loss, actor_loss + self.critic_discount * critic_loss


    def fit(self, envs):
        n_threads = len(envs)
        env_xs, env_as = [[] for _ in range(n_threads)], [[] for _ in range(n_threads)]
        env_rs, env_vs = [[] for _ in range(n_threads)], [[] for _ in range(n_threads)]
        env_rnn_states = [np.zeros(self.rnn_sz) for _ in range(n_threads)]

        # Get Observations from all Environments
        observations = [env.reset() for env in envs]                                    # Shape [n_threads, 28, 28, 1]
        done, all_done, t = np.array([False] for _ in range(n_threads)), False, 1

        # Run Episode Loop
        while not all_done:
            # Stack all Observations into a Single Matrix
            step_xs = np.vstack(observations)

            # Get Logits, Policies/Actions, and Values for all Environments in Single Pass
            step_logits, step_ps, step_vs, step_rnn = self.predict(step_xs, 0.5, env_rnn_states)  # TODO Check Dropout!
            step_as = self.act(step_ps)

            # Perform Action in every Environment, Update Observations
            for i, env in enumerate(envs):
                if not done[i]:
                    obs, r, done[i] = env.step(step_as[i], step_logits[i])


