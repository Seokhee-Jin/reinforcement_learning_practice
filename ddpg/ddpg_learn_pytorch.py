# DDPG Actor (tf2 subclassing version: using chain rule to train Actor)
# coded by St.Watermelon
# https://pasus.tistory.com/138
import argparse
import os
os.makedirs("./save_weights/", exist_ok=True)
'''
import gymnasium
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Lambda, concatenate  # concatenate 레이어를 사용!
from tensorflow.python.keras.optimizer_v2.adam import Adam
import tensorflow as tf

from replaybuffer import ReplayBuffer

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", default=0.95)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--buffer_size", default=20000)
    parser.add_argument("--actor_learning_rate", default=0.0001)
    parser.add_argument("--critic_learning_rate", default=0.001)
    parser.add_argument("--tau", default=0.001)

    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

class Actor(Model):
    def __init__(self, action_dim, action_bound):  # 텐서플로는 초기화 인자로 인풋사이즈 받을 필요 없다. 아웃풋 형태만 인자로 받자.
        super(Actor, self).__init__()

        self.action_bound = action_bound

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.action = Dense(action_dim, activation='tanh') #-1,1로 한 후 스케일 조정

    def call(self, state):  # call 할때 inputs를 받는다.
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        a = self.action(x)

        # Scale output to [-action_bound, action_bound]
        a = Lambda(lambda x: x * self.action_bound)(a)

        return a

class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.x1 = Dense(32, activation='relu')
        self.a1 = Dense(32, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.q = Dense(1, activation='linear')

    def call(self, state_action): #
        state = state_action[0]
        action = state_action[1]
        x = self.x1(state)
        a = self.a1(action)
        h = concatenate([x, a], axis=-1)  # axis는 별 의미 없는듯. 배치 차원만 안거드리면..
        x = self.h2(h)  # 텐서플로에선 굳이 flatten이 필요 없나봄..? 아니면 어차피 1차원이기 때문에 상관없는건가.
        x = self.h3(x)
        q = self.q(x)

        return q

class DDPGagent(object):
    def __init__(self, env: gymnasium.Env):
        args = get_args()

        # hyper parameters
        self.GAMMA = args.gamma
        self.BATCH_SIZE = args.batch_size
        self.BUFFER_SIZE = args.buffer_size
        self.ACTOR_LEARNING_RATE = args.actor_learning_rate
        self.CRITIC_LEARNING_RATE = args.critic_learning_rate
        self.TAU = args.tau

        self.env = env
        # get state dimension
        self.state_dim = env.observation_space.shape[0]
        # get action dimension
        self.action_dim = env.action_space.shape[0]
        # get action bound
        self.action_bound = env.action_space.high[0]

        # create actor and critic networks
        self.actor = Actor(self.action_dim, self.action_bound)
        self.target_actor = Actor(self.action_dim, self.action_bound)  # todo: 흐음 과연 이후 코드에서 어떻게 점진적 업데이트할까..

        self.critic = Critic()
        self.target_critic = Critic()  # todo: 이것도 점진적으로 업데이트 해야할텐데...

        self.actor.build(input_shape=(None, self.state_dim))
        self.target_actor.build(input_shape=(None, self.state_dim))  # todo: 이것도 지워도 될듯. 빌드를 지금 안하면 train할때마다 구조가 바뀔까?


        state_in = Input((self.state_dim,))
        action_in = Input((self.action_dim,))
        self.critic([state_in, action_in]) # todo: 일종의 build인듯. 이럴거면 그냥 Model 첫번째 레이어에 input shape를 지정하지..
        self.target_critic([state_in, action_in])  # todo: 뭔가 일관적이지 않은 사용법때문에 혼동만됨... 테스트해보자


        self.actor.summary()
        self.critic.summary()

        # optimizer
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        # initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # save the results
        self.save_epi_reward = []

    def update_target_network(self, TAU): # 타우는 매우 작은 수.. 이동 평균으로 점진적으로 업데이트하기 위한 함수.
        """ transfer actor weights to target actor with a tau """
        theta = self.actor.get_weights()
        target_theta = self.target_actor.get_weights() # target은 최적화할 땐 동결돼있던 녀석.
        for i in range(len(theta)): # get_weights 해서 순서대로 접근해서 업데이트...
            target_theta[i] = TAU * theta[i] + (1 - TAU) * target_theta[i]
        self.target_actor.set_weights(target_theta)

        phi = self.critic.get_weights()
        target_phi = self.target_critic.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1-TAU) * target_phi[i]
        self.target_critic.set_weights(target_phi)
        # -> 파라미터 업데이트를 그래디언트가 아닌 이동 평균으로 구한다는 독특한 알고리즘!

    def critic_learn(self, states, actions, td_targets):
        """single gradient update on a single batch data"""
        with tf.GradientTape() as tape:
            q = self.critic([states, actions], training=True)
            loss = tf.reduce_mean(tf.square(q-td_targets))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        # -> 아하. loss에 대한 트레이너블 변수였으면 동결 안됨. 이런식으로 동결..
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

    def actor_learn(self, states):  # actor_learn에선 target_actor가 안쓰임. target_actor, target_critic 모두 citic_learn에서 쓰임
        """train the actor network"""
        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            critic_q = self.critic([states, actions])
            loss = -tf.reduce_mean(critic_q)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        """ Ornstein Uhlenbeck Noise """
        return x + rho * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)
        # todo: 가우시안 화이트 노이즈도 사용할 수 있다고 하니 비교테스트해보자.

    def gaussian_noise(self, x, mu=0, sigma=0.2):
        return x + np.random.normal(0, 0.2)

    def td_target(self, rewards, q_values, dones, truncateds):
        """ computing TD target: y_k = r_k + gamma*Q(x_k+1, u_k+1) """
        y_k = np.asarray(q_values)  # todo: np.asarray() 확인해보기...
        for i in range(q_values.shape[0]):  # number of batch
            if dones[i] or truncateds[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k

    def load_weights(self, path):
        self.actor.load_weights(path, "pendulum_actor.h5")
        self.critic.load_weights(path, "pendulum_critic.h5")

    def train(self, max_episode_num):
        #  initial transfer model weights to target model network
        self.update_target_network(1.0) # TAU=1.0 즉 완전복사.

        for ep in range(int(max_episode_num)):
            # reset OU noise
            pre_noise = np.zeros(self.action_dim) # 왜 노이즈를 미리 초기화해놓지? -> ou noise의 정의 자체가 누적되는 값임..
            # reset episode
            time, episode_reward, done, truncated = 0, 0, False, False
            # reset the environment and observe the first state
            state, _ = self.env.reset()

            while not done and not truncated:
                # visualize the environment
                #self.env.render()
                # pick an action
                action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32)) # tensor연산함수이므로 tensor로 바꾸고 배치차원 추가
                action = action.numpy()[0]  # 넘파이 타입으로 바꾸고 배치차원 제거
                noise = self.ou_noise(pre_noise, dim=self.action_dim)
                # clip continuous action to be within action_bound
                action = np.clip(action + noise, -self.action_bound, self.action_bound) # noise가 더해져서 바운드를 넘어갈 수 있으므로 클립.
                # observe reward, new_state
                next_state, reward, done, truncated, _ = self.env.step(action)

                # add transition to replay buffer
                train_reward = (reward + 8) / 8

                self.buffer.add_buffer(state, action, train_reward, next_state, done, truncated)

                if self.buffer.buffer_count() > 1000: # start train after buffer has some amounts

                    # sample transitions from replay buffer
                    states, actions, rewards, next_states, dones, truncateds = self.buffer.sample_batch(self.BATCH_SIZE)

                    # predict target Q-values
                    target_qs = self.target_critic([tf.convert_to_tensor(next_states, tf.float32),
                                                    self.target_actor(tf.convert_to_tensor(next_states, tf.float32))])
                    # -> target_qs 구할때만 target_critic, target_actor가 쓰인다.

                    # compute TD targets
                    y_i = self.td_target(rewards, target_qs.numpy(), dones, truncateds)

                    # train critic usin sampled batch
                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(actions, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))

                    # train actor
                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32))
                    # updates both target network
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

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
'''



