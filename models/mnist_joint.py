"""
mnist.py

Distributed Knowledge Aggregation Network with MNIST CNN Architecture.
"""
from itertools import chain
from keras.layers import Conv2D, Dense, MaxPooling2D
import numpy as np
import tensorflow as tf


class MnistKANJoint(object):
    def __init__(self, supervised_exploit, img_rows=28, img_cols=28, cnn_depths=(32, 64), cnn_filters=((3, 3), (3, 3)),
                 pool_dim=(2, 2), hidden_sz=128, rnn_sz=64, submodule_sz=64, num_classes=10, num_actions=2, max_len=5,
                 critic_discount=0.5, gamma=0.99, lambda_=1.0):
        """
        Initialize an MNIST Knowledge Aggregation Network, with the necessary hyperparameters.
        """
        self.supervised_exploit = supervised_exploit
        self.img_rows, self.img_cols, self.num_classes, self.num_actions = img_rows, img_cols, num_classes, num_actions
        self.cnn_depths, self.cnn_filters, self.pool_dim = cnn_depths, cnn_filters, pool_dim
        self.rnn_sz, self.hidden_sz, self.submodule_sz, self.max_len = rnn_sz, hidden_sz, submodule_sz, max_len
        self.critic_discount, self.gamma, self.lambda_ = critic_discount, gamma, lambda_
        self.session = tf.Session()

        # Setup Supervised Learning Placeholders
        self.Image_Spread = tf.placeholder(tf.float32, shape=[None, self.max_len, self.img_rows, self.img_cols, 1],
                                           name='Full_Observations')
        self.Spread_Length = tf.placeholder(tf.int64, shape=[None])
        self.Label = tf.placeholder(tf.int64, shape=[None], name='True_Label')
        self.Dropout = tf.placeholder(tf.float32, name='Dropout_Probability')

        # Setup Actor-Critic Placeholders
        self.Single_Image = tf.placeholder(tf.float32, shape=[None, self.img_rows, self.img_cols, 1],
                                           name='Single_Sample')
        self.Action = tf.placeholder(tf.float32, shape=[None, self.num_actions], name='Action_Taken')
        self.Reward = tf.placeholder(tf.float32, shape=[None, 1], name='Reward')
        self.Advantage = tf.placeholder(tf.float32, shape=[None, 1], name='Advantage')

        # Setup RNN State Placeholder
        self.RNN_State = tf.placeholder(tf.float32, shape=[None, self.rnn_sz], name='Initial_RNN_State')

        # Initialize Network Weights
        self.instantiate_weights()

        # Compute supervised logits
        #self.super_logits = self.supervised_forward()

        # Compute logits, policy, and value estimate
        self.ac_logits, self.policy, self.value = self.ac_forward()
        self.ac_probabilities = tf.nn.softmax(self.ac_logits)

        # Compute supervised loss, actor-critic loss
        self.supervised_loss, self.ac_loss = self.loss()

        # Build Accuracy Operation
        #correct = tf.equal(tf.argmax(self.super_logits, 1), self.Label)
        #self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="Accuracy")

        # Build Separate Training Operations for Losses
        #self.supervised_train_op = tf.train.AdamOptimizer().minimize(self.supervised_loss)
        self.ac_train_op = tf.train.AdamOptimizer().minimize(self.ac_loss)

        # Initialize all Variables
        self.session.run(tf.global_variables_initializer())

    def instantiate_weights(self):
        self.conv1 = Conv2D(self.cnn_depths[0], kernel_size=self.cnn_filters[0], activation='relu')
        self.conv2 = Conv2D(self.cnn_depths[1], kernel_size=self.cnn_filters[1], activation='relu')
        self.pool = MaxPooling2D(pool_size=self.pool_dim)
        self.hidden = Dense(128, activation='relu')
        self.l_hidden = Dense(self.submodule_sz, activation='relu')
        self.logit_layer = Dense(self.num_classes, activation='linear')
        self.p_hidden = Dense(self.submodule_sz, activation='relu')
        self.policy_layer = Dense(self.num_actions, activation='softmax')
        self.v_hidden = Dense(self.submodule_sz, activation='relu')
        self.value_layer = Dense(1, activation='linear')
        self.cell = tf.contrib.rnn.GRUCell(self.rnn_sz)

    # def supervised_forward(self):
    #     # Reshape into individual images
    #     images = tf.reshape(self.Image_Spread, shape=[-1, self.img_rows, self.img_cols, 1])
    #
    #     # First Convolutional Layer
    #     conv1 = self.conv1(images)                                                 # Shape: [None * max_len, 24, 24, 32]
    #
    #     pool1 = self.pool(conv1)
    #
    #     # Second Convolutional Layer
    #     conv2 = self.conv2(pool1)                                                  # Shape: [None * max_len, 12, 12, 64]
    #
    #     # Pooling Layer
    #     pool2 = self.pool(conv2)                                                   # Shape: [None * max_len, 5, 5, 64]
    #
    #     # Flatten + Dropout
    #     flat_dropout = tf.nn.dropout(tf.reshape(pool2, shape=[-1, 5 * 5 * 64]), self.Dropout)
    #
    #     # Dense Layer
    #     hidden = self.hidden(flat_dropout)                                        # Shape: [None * max_len, hidden]
    #
    #     # Reshape into Time Series
    #     hidden = tf.reshape(hidden, shape=[-1, self.max_len, self.hidden_sz])     # Shape: [None, max_len, hidden]
    #
    #     # GRU - RNN with Empty Initial State
    #     with tf.variable_scope('RNN', reuse=None):
    #         _, self.super_final_state = tf.nn.dynamic_rnn(self.cell, hidden, dtype=tf.float32,
    #                                                       sequence_length=self.Spread_Length)
    #
    #     # Logits Hidden Layer => Linear
    #     l_hidden = self.l_hidden(self.super_final_state)
    #     logits = self.logit_layer(l_hidden)
    #
    #     return logits

    def ac_forward(self):
        # First Convolutional Layer
        conv1 = self.conv1(self.Single_Image)

        pool1 = self.pool(conv1)

        # Second Convolutional Layer
        conv2 = self.conv2(pool1)

        # Pooling Layer
        pool2 = self.pool(conv2)

        # Flatten + Dropout
        flat_dropout = tf.nn.dropout(tf.reshape(pool2, shape=[-1, 5 * 5 * 64]), self.Dropout)

        # Dense Layer
        hidden = self.hidden(flat_dropout)
        hidden = tf.expand_dims(hidden, axis=1)                                              # Shape: [None, 1, hidden]

        # GRU - RNN seeded with Input Initial State
        with tf.variable_scope('RNN', reuse=True):
            _, self.ac_final_state = tf.nn.dynamic_rnn(self.cell, hidden,
                                                       initial_state=self.RNN_State)             # Shape: [None, rnn_sz]

        # Logits Hidden Layer => Linear
        l_hidden = self.l_hidden(self.ac_final_state)
        logits = self.logit_layer(l_hidden)

        # Policy Hidden Layer => Softmax
        p_hidden = self.p_hidden(self.ac_final_state)
        policy = self.policy_layer(p_hidden)

        # Value Hidden Layer => Linear
        v_hidden = self.v_hidden(self.ac_final_state)
        value = self.value_layer(v_hidden)

        return logits, policy, value

    def predict(self, images, dropout, rnn_states):
        return self.session.run([self.ac_probabilities, self.policy, self.value, self.ac_final_state],
                                feed_dict={self.Single_Image: images, self.RNN_State: rnn_states,
                                           self.Dropout: dropout})

    def act(self, policies, episode_no):
        # If Supervised Learning => Take Argmax of Policy
        if episode_no % 2 == 0 or episode_no > self.supervised_exploit:
            return [np.argmax(p) for p in policies]
        # If Reinforcement Learning => Sample from Policy
        else:
            return [np.random.choice(self.num_actions, p=p) for p in policies]

    def loss(self):
        # Supervised Learning Loss
        supervised_loss = tf.losses.sparse_softmax_cross_entropy(self.Label, self.super_logits)

        # Policy Gradient (Actor) Loss
        actor_loss = -tf.reduce_sum(tf.log(self.policy) * self.Action * self.Advantage)

        # Value Function (Critic) Loss
        critic_loss = 0.5 * tf.reduce_sum(tf.square((self.value - self.Reward)))

        return supervised_loss, actor_loss + self.critic_discount * critic_loss

    def train_step(self, env_xs, env_as, env_rs, env_vs, env_rnn_states, env_labels, episode_no):
        # If Episode is Even => Do Supervised Train Step

        # Flatten Observations into 2D Tensor
        xs = np.array(list(chain.from_iterable(env_xs)))

        # Flatten RNN States
        rnn_states = np.vstack(list(chain.from_iterable(map(lambda x: x[:-1], env_rnn_states))))

        # One-Hot Actions
        as_ = np.zeros((len(xs), self.num_actions))
        as_[np.arange(len(xs)), list(chain.from_iterable(env_as))] = 1

        # Compute Discounted Rewards + Advantages
        drs, advs = [], []
        for i in range(len(env_vs)):
            # Compute discounted rewards with a 'bootstrapped' final value
            rs_bootstrap = [] if env_rs[i] == [] else env_rs[i] + [env_vs[i][-1]]
            drs.extend(self._discount(rs_bootstrap, self.gamma)[:-1])

            # Compute advantages via GAE - Schulman et. al. 2016
            delta_t = env_rs[i] + self.gamma * np.array(env_vs[i][1:]) - np.array(env_vs[i][:-1])  # (Eq 11)
            advs.extend(self._discount(delta_t, self.gamma * self.lambda_))

        # Expand drs, advs to be 2D Tensors
        drs, advs = np.array(drs)[:, np.newaxis], np.array(advs)[:, np.newaxis]

        # RL Training Update
        self.session.run(self.ac_train_op, feed_dict={self.Single_Image: xs, self.Action: as_, self.Reward: drs,
                                                      self.Advantage: advs, self.RNN_State: rnn_states,
                                                      self.Dropout: 1.0})

    @staticmethod
    def _discount(x, gamma):
        return [sum(gamma ** i * r for i, r in enumerate(x[t:])) for t in range(len(x))]

    def fit(self, envs, episode_no):
        n_threads = len(envs)
        env_xs, env_as = [[] for _ in range(n_threads)], [[] for _ in range(n_threads)]
        env_rs, env_vs = [[] for _ in range(n_threads)], [[] for _ in range(n_threads)]
        env_rnn_states, env_labels = [[np.zeros(self.rnn_sz)] for _ in range(n_threads)], [[] for _ in range(n_threads)]
        env_rnn_last = [np.zeros(self.rnn_sz) for _ in range(n_threads)]
        episode_rs = np.zeros(len(envs), dtype=np.float)

        # Get Observations from all Environments
        observations = [env.reset() for env in envs]                                    # Shape [n_threads, 28, 28, 1]
        labels = [env.label for env in envs]                                            # Shape [n_threads]
        done, all_done, t = np.array([False for _ in range(n_threads)]), False, 1

        # Run Episode Loop
        while not all_done:
            # Stack all Observations into a Single Matrix
            step_xs = np.array(observations)

            # Get Logits, Policies/Actions, and Values for all Environments in Single Pass
            step_logits, step_ps, step_vs, step_rnn = self.predict(step_xs, 1.0, env_rnn_last)   # TODO Check Dropout!
            step_as = self.act(step_ps, episode_no)

            # Perform Action in every Environment, Update Observations
            for i, env in enumerate(envs):
                if not done[i]:
                    obs, r, done[i] = env.step(step_as[i], step_logits[i])

                    # Record the observation, action, value, reward, and rnn_state in the buffers.
                    env_xs[i].append(step_xs[i])
                    env_as[i].append(step_as[i])
                    env_vs[i].append(step_vs[i][0])
                    env_rs[i].append(r)
                    env_labels[i].append(labels[i])
                    env_rnn_states[i].append(step_rnn[i])
                    env_rnn_last[i] = step_rnn[i]
                    episode_rs[i] += r

                    # Add 0 as the state value when done.
                    if done[i]:
                        env_vs[i].append(0.0)
                    else:
                        observations[i] = obs

                all_done = np.all(done)
                t += 1

        # Perform update when all episodes are finished
        if len(env_xs[0]) > 0:
            self.train_step(env_xs, env_as, env_rs, env_vs, env_rnn_states, env_labels, episode_no)

        return episode_rs

    def eval(self, test_envs, num=500):
        correct, episode_lengths = 0.0, []
        picks = np.random.choice(len(test_envs), num, replace=False)
        for idx in range(num):
            env = test_envs[picks[idx]]
            obs, rnn_state, counter, class_p, done = env.reset(), np.zeros(self.rnn_sz), 0, None, False
            while not done:
                counter += 1
                class_p, policy, _, new_rnn_state = self.predict([obs], 1.0, [rnn_state])
                class_p, policy, rnn_state = class_p[0], policy[0], new_rnn_state[0]

                if np.argmax(policy) == 1:
                    done = True
                else:
                    obs, r, done = env.step(0, class_p)

            # Update Correct, Episode Lengths
            if np.argmax(class_p) == env.label:
                correct += 1
            episode_lengths.append(counter)

        # Return Accuracy, Average Episode Length
        return correct / float(num), np.mean(episode_lengths)
