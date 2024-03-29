# DDPG Actor (tf2 subclassing version: using chain rule to train Actor)
# coded by St.Watermelon
# https://pasus.tistory.com/138
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Lambda, concatenate
from tensorflow.python.keras.optimizer_v2.adam import Adam
import tensorflow as tf

from replaybuffer import ReplayBuffer


# actor network
class Actor(Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_bound = action_bound

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.action = Dense(action_dim, activation='tanh')


    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        a = self.action(x)

        # Scale output to [-action_bound, action_bound]
        a = Lambda(lambda x: x*self.action_bound)(a)

        return a


# critic network
class Critic(Model):

    def __init__(self):
        super(Critic, self).__init__()

        self.x1 = Dense(32, activation='relu')
        self.a1 = Dense(32, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.q = Dense(1, activation='linear')

    def call(self, state_action):
        state = state_action[0]
        action = state_action[1]
        x = self.x1(state)
        a = self.a1(action)
        h = concatenate([x, a], axis=-1)
        x = self.h2(h)
        x = self.h3(x)
        q = self.q(x)
        return q


class DDPGagent(object):

    def __init__(self, env):

        ## hyperparameters
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001

        self.env = env
        # get state dimension
        self.state_dim = env.observation_space.shape[0]
        # get action dimension
        self.action_dim = env.action_space.shape[0]
        # get action bound
        self.action_bound = env.action_space.high[0]

        ## create actor and critic networks
        self.actor = Actor(self.action_dim, self.action_bound)
        self.target_actor = Actor(self.action_dim, self.action_bound)

        self.critic = Critic()
        self.target_critic = Critic()

        self.actor.build(input_shape=(None, self.state_dim))
        self.target_actor.build(input_shape=(None, self.state_dim))

        state_in = Input((self.state_dim,))
        action_in = Input((self.action_dim,))
        self.critic([state_in, action_in])
        self.target_critic([state_in, action_in])

        self.actor.summary()
        self.critic.summary()

        # optimizer
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        ## initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # save the results
        self.save_epi_reward = []


    ## transfer actor weights to target actor with a tau
    def update_target_network(self, TAU):
        theta = self.actor.get_weights()
        target_theta = self.target_actor.get_weights()
        for i in range(len(theta)):
            target_theta[i] = TAU * theta[i] + (1 - TAU) * target_theta[i]
        self.target_actor.set_weights(target_theta)

        phi = self.critic.get_weights()
        target_phi = self.target_critic.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_critic.set_weights(target_phi)


    ## single gradient update on a single batch data
    def critic_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            q = self.critic([states, actions], training=True)
            loss = tf.reduce_mean(tf.square(q-td_targets))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))


    ## train the actor network
    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            critic_q = self.critic([states, actions])
            loss = -tf.reduce_mean(critic_q)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))


    ## Ornstein Uhlenbeck Noise
    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho*(mu - x)*dt + sigma*np.sqrt(dt)*np.random.normal(size=dim)


    ## computing TD target: y_k = r_k + gamma*Q(x_k+1, u_k+1)
    def td_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k


    ## load actor weights
    def load_weights(self, path):
        self.actor.load_weights(path + 'pendulum_actor.h5')
        self.critic.load_weights(path + 'pendulum_critic.h5')


    ## train the agent
    def train(self, max_episode_num):

        # initial transfer model weights to target model network
        self.update_target_network(1.0)

        for ep in range(int(max_episode_num)):
            # reset OU noise
            pre_noise = np.zeros(self.action_dim)
            # reset episode
            time, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            state = self.env.reset()

            while not done:
                # visualize the environment
                #self.env.render()
                # pick an action: shape = (1,)
                action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
                action = action.numpy()[0]
                noise = self.ou_noise(pre_noise, dim=self.action_dim)
                # clip continuous action to be within action_bound
                action = np.clip(action + noise, -self.action_bound, self.action_bound)
                # observe reward, new_state
                next_state, reward, done, _ = self.env.step(action)
                # add transition to replay buffer
                train_reward = (reward + 8) / 8

                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                if self.buffer.buffer_count() > 1000:  # start train after buffer has some amounts

                    # sample transitions from replay buffer
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    # predict target Q-values
                    target_qs = self.target_critic([tf.convert_to_tensor(next_states, dtype=tf.float32),
                                                    self.target_actor(
                                                        tf.convert_to_tensor(next_states, dtype=tf.float32))])
                    # compute TD targets
                    y_i = self.td_target(rewards, target_qs.numpy(), dones)

                    # train critic using sampled batch
                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(actions, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))

                    # train actor
                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32))
                    # update both target network
                    self.update_target_network(self.TAU)

                # update current state
                pre_noise = noise
                state = next_state
                episode_reward += reward
                time += 1

            ## display rewards every episode
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)


            ## save weights every episode
            #print('Now save')
            self.actor.save_weights("./save_weights/pendulum_actor.h5")
            self.critic.save_weights("./save_weights/pendulum_critic.h5")

        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## save them to file if done
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()


