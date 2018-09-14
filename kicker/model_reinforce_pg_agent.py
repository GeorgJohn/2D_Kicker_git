import numpy as np
import tensorflow as tf
import random

slim = tf.contrib.slim


class PGAgent(object):

    def __init__(self, session, state_size, num_actions, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4,
                 hidden_size_5, learning_rate=1e-3, explore_exploit_setting='epsilon_greedy_0.05'):
        self.session = session
        self.state_size = state_size
        self.num_actions = num_actions
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.hidden_size_4 = hidden_size_4
        self.hidden_size_5 = hidden_size_5
        self.learning_rate = learning_rate
        self.explore_exploit_setting = explore_exploit_setting

        self.build_model()
        self.build_training()
        self.saver = tf.train.Saver()

    def build_model(self):
        with tf.variable_scope('pg-model'):
            self.state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
            self.h0 = slim.fully_connected(self.state, self.hidden_size_1, activation_fn=tf.nn.relu)
            self.h1 = slim.fully_connected(self.h0, self.hidden_size_2, activation_fn=tf.nn.relu)
            self.h2 = slim.fully_connected(self.h1, self.hidden_size_3, activation_fn=tf.nn.relu)
            self.h3 = slim.fully_connected(self.h2, self.hidden_size_4, activation_fn=tf.nn.relu)
            self.h4 = slim.fully_connected(self.h3, self.hidden_size_5, activation_fn=tf.nn.relu)
            self.output = slim.fully_connected(self.h4, self.num_actions, activation_fn=tf.nn.softmax)

    def build_training(self):
        self.action_input = tf.placeholder(tf.int32, shape=[None])
        self.reward_input = tf.placeholder(tf.float32, shape=[None])

        # Select the logits related to the action taken
        self.output_index_for_actions = (tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1]) + \
                                        self.action_input
        self.logits_for_actions = tf.gather(tf.reshape(self.output, [-1]), self.output_index_for_actions)
        self.loss = - tf.reduce_mean(tf.log(tf.clip_by_value(self.logits_for_actions, 1e-32, 1.0)) * self.reward_input)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)

    def sample_action_from_distribution(self, action_distribution, epsilon_percentage):

        # Choose an action based on the action probability
        # distribution and an explore vs exploit
        if self.explore_exploit_setting == 'no_greedy':
            action = np.argmax(action_distribution)
        elif self.explore_exploit_setting == 'greedy':
            action = self.epsilon_greedy_action(action_distribution)
        elif self.explore_exploit_setting == 'epsilon_greedy_0.05':
            action = self.epsilon_greedy_action(action_distribution, 0.05)
        elif self.explore_exploit_setting == 'epsilon_greedy_0.25':
            action = self.epsilon_greedy_action(action_distribution, 0.25)
        elif self.explore_exploit_setting == 'epsilon_greedy_0.50':
            action = self.epsilon_greedy_action(action_distribution, 0.50)
        elif self.explore_exploit_setting == 'epsilon_greedy_0.90':
            action = self.epsilon_greedy_action(action_distribution, 0.90)
        elif self.explore_exploit_setting == 'epsilon_greedy_annealed_1.0->0.001':
            action = self.epsilon_greedy_action_annealed(action_distribution, epsilon_percentage, 1.0, 0.001)
        elif self.explore_exploit_setting == 'epsilon_greedy_annealed_0.5->0.001':
            action = self.epsilon_greedy_action_annealed(action_distribution, epsilon_percentage, 0.5, 0.001)
        elif self.explore_exploit_setting == 'epsilon_greedy_annealed_0.25->0.001':
            action = self.epsilon_greedy_action_annealed(action_distribution, epsilon_percentage, 0.25, 0.001)
        else:
            raise Exception('Please choose a greedy method')
        return action

    def predict_action(self, state, epsilon_percentage):
        action_distribution = self.session.run(self.output, feed_dict={self.state: [state]})[0]
        # action = np.argmax(action_distribution)
        # print(action_distribution)
        action = self.sample_action_from_distribution(action_distribution, epsilon_percentage)
        return action

    @staticmethod
    def epsilon_greedy_action(action_distribution, epsilon=1e-1):
        if random.random() < epsilon:
            return np.argmax(np.random.random(action_distribution.shape))
        else:
            return np.argmax(action_distribution)

    @staticmethod
    def epsilon_greedy_action_annealed(action_distribution, percentage, epsilon_start=1.0, epsilon_end=1e-2):
        annealed_epsilon = epsilon_start * (1.0 - percentage) + epsilon_end * percentage
        if random.random() < annealed_epsilon:
            return np.argmax(np.random.random(action_distribution.shape))
        else:
            return np.argmax(action_distribution)
