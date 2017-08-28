from environments.identity_mnist import IdentityMnist
from keras.datasets import mnist
import tensorflow as tf
import numpy as np

MNIST_EXPERIMENT = "Identity"
N_THREADS = 8
NEED_MORE_INFO = 10

def leaky_relu(x, alpha=0.001):
    return tf.maximum(x, alpha*x)


def encode_image(image, name, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        c1 = tf.layers.conv2d(image, 16, 5, activation=leaky_relu, strides=2, name='c1')    # 12 x 12 x 16
        c2 = tf.layers.conv2d(c1, 32, 5, activation=leaky_relu, strides=2, name='c2')       # 4 x 4 x 32
        fc1 = tf.layers.dense(tf.reshape(c2, [-1, 4*4*32]), 100, activation=leaky_relu, name='fc1')
    return fc1


def to_policy_vector(hidden_state, num_classes, name, hidden_sz=128, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        hidden = tf.layers.dense(hidden_state, hidden_sz, activation=leaky_relu, name='hidden')
        out = tf.layers.dense(hidden, num_classes + 1, activation=tf.nn.softmax, name='policy')
    return out


def to_value(hidden_state, name, hidden_sz=128, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        hidden = tf.layers.dense(hidden_state, hidden_sz, activation=leaky_relu, name='hidden')
        out = tf.layers.dense(hidden, 1, activation=None, name='value')
    return out


class Agent(object):
    def __init__(self, max_iters=5, num_classes=10, critic_discount=0.5, gamma=0.99, lambda_=1.0):
        self.max_iters, self.num_classes = max_iters, num_classes
        self.critic_discount, self.gamma, self.lambda_ = critic_discount, gamma, lambda_
        self.build_network()
        self.sess = tf.Session()

        # Initialize all Variables
        self.sess.run(tf.global_variables_initializer())

    def build_network(self):
        self.inp_images = tf.placeholder(tf.float32, [None, self.max_iters, 28, 28, 1])
        self.inp_selected_time = tf.placeholder(tf.int32, [None])
        self.inp_selected_action = tf.placeholder(tf.int32, [None])
        self.inp_reward = tf.placeholder(tf.float32, [None])
        self.inp_advantage = tf.placeholder(tf.float32, [None])
        self.cell = tf.contrib.rnn.GRUCell(128)

        ############
        # This part of the code creates the policy vector of size [batch_size, max_iters, 2*num_classes]
        hidden_inputs = []
        for i in range(self.max_iters):
            encoding = encode_image(self.inp_images[:, i, :, :, :], 'encoder', reuse=i > 0)
            hidden_inputs.append(encoding)
        hidden_inputs = tf.transpose(hidden_inputs, [1, 0, 2])              # [batch_size, max_iters, hidden_size]

        # GRU - RNN with Empty Initial State
        rnn_outputs, _ = tf.nn.dynamic_rnn(self.cell, hidden_inputs, dtype=tf.float32, scope='rnn')
        all_policies = []
        for i in range(self.max_iters):
            policy = to_policy_vector(rnn_outputs[:, i, :], self.num_classes, 'policy', reuse=i > 0)
            all_policies.append(policy)
        self.all_policies = tf.transpose(all_policies, [1, 0, 2])           # [bs, max_iters, 2*num_classes]

        all_values = []
        for i in range(self.max_iters):
            value = to_value(rnn_outputs[:, i, :], 'value', reuse=i > 0)
            all_values.append(value)
        self.all_values = tf.transpose(all_values, [1, 0, 2])               # [bs, max_iters, 1]

        # This part of the code selects the policy using the first "accept" policy.
        time_selector = tf.reshape(tf.one_hot(self.inp_selected_time, self.max_iters), [-1, self.max_iters, 1])
        self.selected_policy = tf.reduce_sum(self.all_policies * time_selector, axis=1)  # [bs, 2*num_classes]
        self.selected_value = tf.squeeze(tf.reduce_sum(self.all_values * time_selector, axis=1))     # [bs]

        # Get actual selected policy
        action_selector = tf.one_hot(self.inp_selected_action, self.num_classes + 1)
        self.policy_probability = tf.reduce_sum(self.selected_policy * action_selector, axis=1)   # [bs]

        ############
        # Actor-Critic Loss Computation
        actor_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(self.policy_probability, 1e-10, 1.0)) * self.inp_advantage)
        critic_loss = 0.5 * tf.reduce_sum(tf.square((self.selected_value - self.inp_reward)))
        self.loss = actor_loss + self.critic_discount * critic_loss

        # Create Training Operation
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def predict(self, images):
        return self.sess.run([self.all_policies, self.all_values], feed_dict={self.inp_images: images})

    def train_step(self, images, times, actions, values, rewards):
        # Compute Discounted Rewards + Advantages
        drs, advs = [], []
        for i in range(len(values)):
            # Compute discounted rewards with a 'bootstrapped' final value.
            rs_bootstrap = [rewards[i]] + [values[i][-1]]
            drs.extend(self._discount(rs_bootstrap, self.gamma)[:-1])

            # Compute advantages via GAE - Schulman et. al 2016
            delta_t = rewards[i] + self.gamma * np.array(values[i][1:]) - np.array(values[i][:-1])  # (Eq 11)
            advs.extend(self._discount(delta_t, self.gamma * self.lambda_))                         # (Eq 16)

        # Perform Training Update
        self.sess.run(self.train_op, feed_dict={self.inp_images: images, self.inp_selected_time: times,
                                                self.inp_selected_action: actions, self.inp_reward: drs,
                                                self.inp_advantage: advs})

    @staticmethod
    def _discount(x, gamma):
        return [sum(gamma ** i * r for i, r in enumerate(x[t:])) for t in range(len(x))]

    def fit(self, envs):
        # Get Observations from all Environments
        observations = [env.reset() for env in envs]                   # Shape [n_threads, max_iters, 28, 28, 1]
        episode_rs = np.zeros(len(envs), dtype=np.float)

        # Get all Policies
        policies, values = self.predict(observations)
        selected_times, selected_actions, state_values, rewards = [], [], [], []
        for idx in range(len(policies)):
            # Sampling Loop
            for j in range(self.max_iters - 1):
                action = np.random.choice(11, p=policies[idx][j])
                if action < 10:
                    selected_actions.append(action)
                    selected_times.append(j)
                    state_values.append([values[idx][j][0], 0.0])
                    rewards.append(envs[idx].reward(policies[idx][j], j))
                    break

            # If not action taken by last time step => force action by taking argmax of first 10 entries
            if len(selected_actions) != idx + 1:
                selected_times.append(self.max_iters - 1)
                selected_actions.append(np.argmax(policies[idx][self.max_iters - 1][:-1]))
                state_values.append([values[idx][self.max_iters - 1][0], 0.0])
                rewards.append(envs[idx].reward(policies[idx][self.max_iters - 1], self.max_iters - 1))

            episode_rs[idx] += rewards[idx]

        # Run Train Step
        self.train_step(observations, selected_times, selected_actions, state_values, rewards)

        # Return Episode Rewards
        return episode_rs

    def eval(self, test_envs, num=500):
        correct, episode_lengths = 0.0, []
        picks = np.random.choice(len(test_envs), num, replace=False)
        for idx in range(num):
            env = test_envs[picks[idx]]
            obs = env.reset()
            [policies], _ = self.predict([obs])

            for i in range(self.max_iters):
                action = np.argmax(policies[i])
                if action != 10:
                    if action == env.label:
                        correct += 1
                        episode_lengths.append(i)

            if len(episode_lengths) != idx + 1:
                episode_lengths.append(self.max_iters - 1)

        # Return Accuracy, Average Episode Length
        return correct / float(num), np.mean(episode_lengths)


if __name__ == "__main__":
    # Load Datasets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Instantiate Model
    mnist_kan, envs, test_envs = Agent(), None, None

    if MNIST_EXPERIMENT == 'Identity':
        # Create environments
        envs = [IdentityMnist(x_train, y_train) for _ in range(N_THREADS)]
        test_envs = [IdentityMnist(x_test[i], y_test[i], seed=i) for i in range(len(x_test))]

    # Fit Model
    running_reward = None
    for e in range(5000):
        episode_rs = mnist_kan.fit(envs)

        for er in episode_rs:
            running_reward = er if running_reward is None else (0.99 * running_reward + 0.01 * er)

        if e % 10 == 0:
            print 'Batch {:d} (episode {:d}), batch avg. reward: {:.2f}, running reward: {:.3f}' \
                .format(e, (e + 1) * N_THREADS, np.mean(episode_rs), running_reward)

        if e % 500 == 0:
            accuracy, average_length = mnist_kan.eval(test_envs)
            print ''
            print 'Sampled Test Accuracy: %.3f\tAverage # of Observations: %.2f' % (accuracy, average_length)
            print ''

