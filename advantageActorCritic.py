import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class

    def __init__(self, model, lr, critic_model, critic_lr, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model = model
        self.critic_model = critic_model
        self.n = n

		
        self.lr = lr
        self.critic_lr = critic_lr

        self.total_reward_test = []

        self.action_space_size = 4

        self.k_train_period = 200
        self.eval_episodes_length = 100

        self.total_reward_test_mean = []
        self.total_reward_test_std = []

    def train(self, env, gamma=1.0, num_episodes=50000):
        # Trains the model on a single episode using A2C.

        for episode in range(num_episodes):
            states, actions, rewards = self.generate_episode(env)

            G = np.zeros_like(rewards)
            V = self.critic_model.predict(states).squeeze()
            G[-1] = rewards[-1]
            for t in range(len(rewards)-2, -1, -1):
                G[t] = rewards[t] + gamma * G[t+1]
            # end_for
            gamma_n = gamma ** self.n

            # print(np.shape(G))

            if self.n >= len(G):
                G_tmp = np.zeros_like(G)
                V_end = np.zeros_like(G)
            else:
                G_tmp = np.hstack((gamma_n * G[self.n:], np.zeros((self.n,))))
                V_end = gamma_n * np.hstack((V[self.n:], np.zeros((self.n,))))

            # G_new = G - np.hstack((gamma_n * G[self.n:], np.zeros((self.n,))))
            G_new = G - G_tmp

            R = V_end + G_new
            advantage = R-V

            history_actor  = self.model.train_on_batch(states, actions*advantage[:, np.newaxis])
            history_critic = self.critic_model.train_on_batch(states, R)

            if episode % self.k_train_period == 0:
                test_rewards_curr = self.test(env)
                print('Test-Reward at Episode %d: \t %.2f' % (episode, float(np.mean(test_rewards_curr))))
                self.total_reward_test_mean.append(np.mean(test_rewards_curr))
                self.total_reward_test_std.append(np.std(test_rewards_curr))
                plt.figure()
                reward_mean_tmp = np.array(self.total_reward_test_mean)
                reward_std_tmp = np.array(self.total_reward_test_std)
                plt.plot(range(len(reward_mean_tmp)), reward_mean_tmp)
                plt.fill_between(range(len(reward_mean_tmp)),
                                 reward_mean_tmp-reward_std_tmp,
                                 reward_mean_tmp+reward_std_tmp, alpha=0.5)
                plt.xlabel('Iterations')
                plt.ylabel('Average Test Reward')
                plt.savefig('A2C-Reward_tmp.png')

                np.save('Total_A2C_MeanTest_Reward_tmp', reward_mean_tmp)
                np.save('Total_A2C_StdTest_Reward_tmp', reward_std_tmp)
            # end_if
        # end_for
        return

    def test(self, env):

        rewards_total = []
        for episode in range(self.eval_episodes_length):
            reward_net = 0
            state = env.reset()
            is_done = False
            time_step = 0
            while not is_done:
                action_prob = self.model.predict(np.array([state])).squeeze()
                action = np.random.choice(self.action_space_size, 1, p=action_prob)[0]
                next_state, reward, is_done, info = env.step(action)
                reward_net += reward
                if is_done:
                    break
                state = next_state
            # end_while
            rewards_total.append(reward_net)
        # end_for

        return rewards_total

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        states = []
        actions = []
        rewards = []

        done = False
        state = env.reset()

        while not done:
            action_prob = self.model.predict(np.array([state])).squeeze()
            action = np.random.choice(self.action_space_size, 1, p=action_prob)[0]
            states.append(np.array(state))
            actions.append(keras.utils.to_categorical(action, self.action_space_size))
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break
            state = next_state
        # end_while

        return np.array(states), np.array(actions), np.array(rewards)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")
						
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    # args = parse_arguments()
    # num_episodes = args.num_episodes
    # lr = args.lr
    # critic_lr = args.critic_lr
    # n = args.n
    # render = args.render

    num_episodes = 50000
    n = 100
    print(n)

    if n == 1:
        lr = 2e-4
        critic_lr = 5e-4
    elif n == 20:
        lr = 2e-4
        critic_lr = 1e-3
    elif n == 50:
        lr = 5e-4
        critic_lr = 2e-4
    else: # n==100
        lr = 2e-4
        critic_lr = 1e-4


    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Create the model.
    # Actor Model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(16, activation='relu', input_shape=env.observation_space.shape,
              bias_initializer='zeros', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0,
                                                                                             distribution='uniform')))
    model.add(keras.layers.Dense(16, activation='relu', bias_initializer='zeros',
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0,
                                                                                       distribution='uniform')))
    model.add(keras.layers.Dense(16, activation='relu', bias_initializer='zeros',
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0,
                                                                                       distribution='uniform')))
    model.add(keras.layers.Dense(4, activation='softmax', bias_initializer='zeros',
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0,
                                                                                       distribution='uniform')))
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.train.AdamOptimizer(),
                  metrics=['accuracy'])

    # Critic Model
    critic_model = keras.models.Sequential()
    critic_model.add(keras.layers.Dense(16, activation='relu', input_shape=env.observation_space.shape,
                                        bias_initializer='zeros',
                                        kernel_initializer=keras.initializers.VarianceScaling(scale=1.0,
                                                                                              distribution='uniform')))
    critic_model.add(keras.layers.Dense(16, activation='relu', bias_initializer='zeros',
                                        kernel_initializer=keras.initializers.VarianceScaling(scale=1.0,
                                                                                              distribution='uniform')))
    critic_model.add(keras.layers.Dense(16, activation='relu', bias_initializer='zeros',
                                        kernel_initializer=keras.initializers.VarianceScaling(scale=1.0,
                                                                                              distribution='uniform')))
    critic_model.add(keras.layers.Dense(1, activation='linear', # bias_initializer='zeros',
                                        kernel_initializer=keras.initializers.VarianceScaling(scale=1.0,
                                                                                              distribution='uniform')))
    critic_model.compile(loss='mean_squared_error',
                         optimizer=tf.train.AdamOptimizer(),
                         metrics=['accuracy'])

    # Train the model using A2C and plot the learning curves.
    a2c = A2C(model, lr, critic_model, critic_lr, n)

    a2c.train(env, gamma=0.99, num_episodes=num_episodes)

    reward_mean_tmp = np.array(a2c.total_reward_test_mean)
    reward_std_tmp = np.array(a2c.total_reward_test_std)
    np.save('Total_A2C_MeanTest_Reward', reward_mean_tmp)
    np.save('Total_A2C_StdTest_Reward', reward_std_tmp)

    # Final Plot
    plt.figure()

    plt.plot(range(len(reward_mean_tmp)), reward_mean_tmp)
    plt.fill_between(range(len(reward_mean_tmp)),
                     reward_mean_tmp - reward_std_tmp,
                     reward_mean_tmp + reward_std_tmp, alpha=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Average Test Reward')
    plt.savefig('A2C-Reward.png')

    print('Shahriar')

if __name__ == '__main__':
    main(sys.argv)
