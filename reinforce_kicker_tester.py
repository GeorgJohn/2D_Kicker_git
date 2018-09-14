import numpy as np
import tqdm

import kicker.control_environment as Env


def main():
    total_episodes = 100000
    max_episode_length = 3500
    episode_rewards = []
    s_path_reward = "/home/prock/data/kicker_reward_random_tester.txt"

    env = Env.EnvironmentController()

    for i in tqdm.tqdm(range(total_episodes)):
        state, _, _ = env.reset()
        episode_reward = 0.0

        for j in range(max_episode_length):
            action = env.get_random_action()

            _, reward, terminal = env.step(action)

            episode_reward += reward

            if terminal:
                episode_rewards.append(episode_reward)
                break

        if np.mod(i, 1000) == 0 and i > 0:
            print('Mean Reward: ', np.mean(episode_rewards), '  Games: ', i)

    with open(s_path_reward, "w") as fp:
        fp.writelines('Testdaten kicker_pong mit zufälligen Aktionen \n')
        for k in episode_rewards:
            fp.write("%s\n" % k)


if __name__ == '__main__':
    main()
