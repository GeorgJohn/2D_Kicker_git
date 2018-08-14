import numpy as np
import tensorflow as tf
import pygame

import control_environment as env


class PolicyGradientAgent(object):

    def __init__(self, hparams, sess):

        # initialization
        self._s = sess

        # build the graph
        self._input = tf.placeholder(tf.float32, shape=[None, hparams['input_size']])

        hidden1 = tf.contrib.layers.fully_connected(inputs=self._input, num_outputs=hparams['hidden_size_1'],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.random_normal_initializer())
        hidden2 = tf.contrib.layers.fully_connected(inputs=hidden1, num_outputs=hparams['hidden_size_2'],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.random_normal_initializer())
        hidden3 = tf.contrib.layers.fully_connected(inputs=hidden2, num_outputs=hparams['hidden_size_3'],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.random_normal_initializer())
        hidden4 = tf.contrib.layers.fully_connected(inputs=hidden3, num_outputs=hparams['hidden_size_4'],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.random_normal_initializer())
        hidden5 = tf.contrib.layers.fully_connected(inputs=hidden4, num_outputs=hparams['hidden_size_5'],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.random_normal_initializer())
        logits = tf.contrib.layers.fully_connected(inputs=hidden5, num_outputs=hparams['num_actions'],
                                                   activation_fn=None)

        # op to sample an action
        self._sample = tf.reshape(tf.multinomial(logits, 1), [])

        # get log probabilities
        log_prob = tf.log(tf.nn.softmax(logits))

        # training part of graph
        self._acts = tf.placeholder(tf.int32)
        self._advantages = tf.placeholder(tf.float32)

        # get log probs of actions from episode
        indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self._acts
        act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

        # surrogate loss
        loss = -tf.reduce_sum(tf.multiply(act_prob, self._advantages))

        # update
        optimizer = tf.train.RMSPropOptimizer(hparams['learning_rate'])
        self._train = optimizer.minimize(loss)

    def act(self, observation):
        # get one action, by sampling
        return self._s.run(self._sample, feed_dict={self._input: [observation]})

    def train_step(self, obs, acts, advantages):
        batch_feed = {self._input: obs, self._acts: acts, self._advantages: advantages}
        self._s.run(self._train, feed_dict=batch_feed)


def policy_rollout(environment, agent, render):
    """Run one episode."""

    observation, reward, done = environment.reset()
    obs, acts, rews = [], [], []

    while not done:

        if render:
            environment.render()

        obs.append(observation[-6:])

        action = agent.act(observation[-6:])
        print(action)
        observation, reward, done = environment.step(action)

        acts.append(action)
        rews.append(reward)

    return obs, acts, rews


def process_rewards(rews):
    """Rewards -> Advantages for one episode. """

    # total reward: length of episode
    return [len(rews)] * len(rews)


def main():

    environment = env.EnvironmentController()

    # monitor_dir = '/tmp/cartpole_exp1'
    # environment.monitor.start(monitor_dir, force=True)

    # hyper parameters
    hparams = {
        'input_size': 6,
        'hidden_size_1': 300,
        'hidden_size_2': 600,
        'hidden_size_3': 600,
        'hidden_size_4': 600,
        'hidden_size_5': 300,
        'num_actions': 3,
        'learning_rate': 0.001
    }

    # environment params
    eparams = {
        'num_batches': 5000,
        'ep_per_batch': 5,
        'num_show_solution': 100
    }

    with tf.Graph().as_default(), tf.Session() as sess:

        agent = PolicyGradientAgent(hparams, sess)

        sess.run(tf.global_variables_initializer())

        should_render = False

        for batch in range(eparams['num_batches']):

            print('=====\nBATCH {}\n===='.format(batch))

            b_obs, b_acts, b_rews = [], [], []

            for _ in range(eparams['ep_per_batch']):

                obs, acts, rews = policy_rollout(environment, agent, should_render)

                print('Episode steps: {}'.format(len(obs)))

                b_obs.extend(obs)
                b_acts.extend(acts)

                advantages = process_rewards(rews)
                b_rews.extend(advantages)

            # update policy
            # normalize rewards; don't divide by 0
            print(np.mean(b_rews))
            b_rews = (b_rews - np.mean(b_rews)) / (np.std(b_rews) + 1e-10)

            agent.train_step(b_obs, b_acts, b_rews)

        should_render = True
        input()
        clock = pygame.time.Clock()

        for _ in range(eparams['num_show_solution']):
            policy_rollout(environment, agent, should_render)
            clock.tick_busy_loop(30)


if __name__ == "__main__":
    main()
