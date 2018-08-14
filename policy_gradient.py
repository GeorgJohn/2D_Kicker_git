import tensorflow as tf
import numpy as np
import random
import collections
import matplotlib.pyplot as plt

from kicker import plot_reward
from kicker import control_environment as env

slim = tf.contrib.slim

random.seed()


def calculate_naive_returns(rewards):
    """ Calculates a list of naive returns given a
    list of rewards."""
    total_returns = np.zeros(len(rewards))
    total_return = 0.0
    for t in range(len(rewards), 0):
        total_return = total_return + rewards
        total_returns[t] = total_return
    return total_returns


def discount_rewards(rewards, gamma=0.98):
    discounted_returns = [0 for _ in rewards]
    discounted_returns[-1] = rewards[-1]
    for t in range(len(rewards)-2, -1, -1):  # iterate backwards
        discounted_returns[t] = rewards[t] + discounted_returns[t+1]*gamma
    return discounted_returns


def epsilon_greedy_action(action_distribution, epsilon=1e-1):
    if random.random() < epsilon:
        return np.argmax(np.random.random(
           action_distribution.shape))
    else:
        return np.argmax(action_distribution)


def epsilon_greedy_action_annealed(action_distribution, percentage, epsilon_start=1.0, epsilon_end=1e-2):
    annealed_epsilon = epsilon_start*(1.0-percentage) + epsilon_end*percentage
    if random.random() < annealed_epsilon:
        return np.argmax(np.random.random(action_distribution.shape))
    else:
        return np.argmax(action_distribution)


class PGAgent(object):

    def __init__(self, session, state_size, num_actions, hidden_size_1, hidden_size_2, learning_rate=1e-3,
                 explore_exploit_setting='epsilon_greedy_0.05'):
        self.session = session
        self.state_size = state_size
        self.num_actions = num_actions
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.learning_rate = learning_rate
        self.explore_exploit_setting = explore_exploit_setting

        self.build_policy_model()
        self.build_training()
        self.saver = tf.train.Saver()

    def build_policy_model(self):
        with tf.variable_scope('pg-model'):
            self.state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
            self.h0 = slim.fully_connected(self.state, self.hidden_size_1, activation_fn=tf.nn.relu)
            self.h1 = slim.fully_connected(self.h0, self.hidden_size_2, activation_fn=tf.nn.relu)
            self.h2 = slim.fully_connected(self.h1, self.hidden_size_2, activation_fn=tf.nn.relu)
            self.h3 = slim.fully_connected(self.h2, self.hidden_size_2, activation_fn=tf.nn.relu)
            self.h4 = slim.fully_connected(self.h3, self.hidden_size_1, activation_fn=tf.nn.relu)
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
        if self.explore_exploit_setting == 'greedy':
            action = epsilon_greedy_action(action_distribution)
        elif self.explore_exploit_setting == 'epsilon_greedy_0.05':
            action = epsilon_greedy_action(action_distribution, 0.05)
        elif self.explore_exploit_setting == 'epsilon_greedy_0.25':
            action = epsilon_greedy_action(action_distribution, 0.25)
        elif self.explore_exploit_setting == 'epsilon_greedy_0.50':
            action = epsilon_greedy_action(action_distribution, 0.50)
        elif self.explore_exploit_setting == 'epsilon_greedy_0.90':
            action = epsilon_greedy_action(action_distribution, 0.90)
        elif self.explore_exploit_setting == 'epsilon_greedy_annealed_1.0->0.001':
            action = epsilon_greedy_action_annealed(action_distribution, epsilon_percentage, 1.0, 0.001)
        elif self.explore_exploit_setting == 'epsilon_greedy_annealed_0.5->0.001':
            action = epsilon_greedy_action_annealed(action_distribution, epsilon_percentage, 0.5, 0.001)
        elif self.explore_exploit_setting == 'epsilon_greedy_annealed_0.25->0.001':
            action = epsilon_greedy_action_annealed(action_distribution, epsilon_percentage, 0.25, 0.001)
        else:
            action = env.Action.NOOP
        return action

    def predict_action(self, state, epsilon_percentage):
        action_distribution = self.session.run(self.output, feed_dict={self.state: [state]})[0]
        # action = np.argmax(action_distribution)
        # print(action_distribution)
        action = self.sample_action_from_distribution(action_distribution, epsilon_percentage)
        return action


class EpisodeHistory(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []
        self.discounted_returns = []

    def add_to_history(self, state, action, reward, state_prime):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_primes.append(state_prime)


class Memory(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []
        self.discounted_returns = []

    def reset_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []
        self.discounted_returns = []

    def add_episode(self, episode):
        self.states += episode.states
        self.actions += episode.actions
        self.rewards += episode.rewards
        self.discounted_returns += episode.discounted_returns


class RewardsMemory(object):

    def __init__(self):
        self.rewards = []
        self.episodes = []


def main():
    # Configure Settings
    total_episodes = 5000
    total_steps_max = 10000
    epsilon_stop = 5000
    train_frequency = 2
    plot_frequency = 10
    max_episode_length = 20000
    render_start = 2000
    should_render = False
    should_plot = True
    episode_finished = False

    explore_exploit_setting = 'epsilon_greedy_0.05'

    environment = env.EnvironmentController()
    state_size = 7
    num_actions = 7

    solved = False
    save_step_1 = False
    save_step_2 = False
    save_step_3 = False
    save_step_4 = False
    save_step_5 = False
    save_step_6 = False

    with tf.Session() as session:
        agent = PGAgent(session=session, state_size=state_size, num_actions=num_actions, hidden_size_1=700,
                        hidden_size_2=1400, learning_rate=0.0005, explore_exploit_setting=explore_exploit_setting)
        session.run(tf.global_variables_initializer())

        episode_rewards = collections.deque(maxlen=2048)
        batch_losses = collections.deque(maxlen=2048)

        plt.ion()
        reward_memory = RewardsMemory()
        reward_plt = plot_reward.RewardPlot()

        global_memory = Memory()
        steps = 0
        train_it = np.uint(0)
        plot_it = np.uint(0)
        i = np.uint64(0)

        try:
            while not solved:

                # for i in tqdm.tqdm(range(total_episodes)):

                state, _, _ = environment.reset()
                state = state[-state_size:]
                episode_reward = 0.0
                episode_history = EpisodeHistory()
                epsilon_percentage = float(min(i/float(epsilon_stop), 1.0))
                for j in range(max_episode_length):
                    action = agent.predict_action(state, epsilon_percentage)

                    state_prime, reward, terminal = environment.step(action)
                    state_prime = state_prime[-state_size:]
                    if (render_start > 0 and i > render_start and should_render):  # or (solved and should_render):
                        environment.render()
                    episode_history.add_to_history(state, action, reward, state_prime)
                    state = state_prime
                    episode_reward += reward
                    steps += 1
                    if terminal:
                        episode_history.discounted_returns = discount_rewards(episode_history.rewards, 0.99)
                        global_memory.add_episode(episode_history)

                        # print('Discounted Returns', len(global_memory.discounted_returns))
                        # print('Actions', len(global_memory.actions))
                        # print('States', len(global_memory.states))

                        plot_it += 1
                        train_it += 1
                        i += 1

                        if np.mod(train_it, train_frequency) == 0:
                            feed_dict = {agent.reward_input: np.array(global_memory.discounted_returns),
                                         agent.action_input: np.array(global_memory.actions),
                                         agent.state: np.array(global_memory.states)}
                            _, batch_loss = session.run([agent.train_step, agent.loss], feed_dict=feed_dict)

                            print(global_memory.actions)
                            batch_losses.append(batch_loss)
                            global_memory.reset_memory()

                            episode_rewards.append(episode_reward)
                            train_it = 0
                            episode_finished = True

                        if i > plot_frequency + 1:
                            if np.mod(plot_it, plot_frequency) == 0 and should_plot:
                                reward_memory.rewards.append(np.mean(episode_rewards))
                                reward_memory.episodes.append(i)
                                reward_plt.update(reward_memory.episodes, reward_memory.rewards)

                        if episode_finished:
                            break

                if np.mod(i, 100) == 0:
                    print('Mean Reward: ', np.mean(episode_rewards), '  Episodes: ', i)
                    print('Anzahl der gewonnen Spiele: ', environment.get_goal_counter(), '/100 Spielen')
                    environment.set_goal_counter(0)
                    # print('Batch_losses: ', batch_losses)

                if i > 5000:
                    if np.mean(episode_rewards) > 2.0:
                        solved = True
                        save_path = agent.saver.save(session, "/tmp/model_solved.ckpt")
                        print("Model saved in path: %s" % save_path)
                    elif np.mean(episode_rewards) > -5.2 and not save_step_6:
                        save_path = agent.saver.save(session, "/tmp/model_reward_5_2.ckpt")
                        print("Model saved in path: %s" % save_path)
                        save_step_6 = True
                    elif np.mean(episode_rewards) > -5.3 and not save_step_5:
                        save_path = agent.saver.save(session, "/tmp/model_reward_5_3.ckpt")
                        print("Model saved in path: %s" % save_path)
                        save_step_5 = True
                    elif np.mean(episode_rewards) > -5.4 and not save_step_4:
                        save_path = agent.saver.save(session, "/tmp/model_reward_5_4.ckpt")
                        print("Model saved in path: %s" % save_path)
                        save_step_4 = True
                    else:
                        solved = False

        except KeyboardInterrupt:
            print('Solved:', solved, 'Mean Reward', np.mean(episode_rewards), '  Episodes: ', i)
            save_path = agent.saver.save(session, "/tmp/finished_model_without_rack_reward.ckpt")
            print("Model saved in path: %s" % save_path)
            pass


if __name__ == '__main__':
    main()
