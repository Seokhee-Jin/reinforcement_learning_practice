#coded by St.Watermelon

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Lambda
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.activations import relu, tanh, softplus

import gymnasium as gym

import numpy as np
import matplotlib.pyplot as plt


import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae_lambda", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--actor_learning_rate", type=float, default=0.0001)
    parser.add_argument("--critic_learning_rate", type=float, default=0.001)
    parser.add_argument("--ratio_clipping", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=5)

    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]



class Actor(Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound

        self.h1 = Dense(64, activation=relu)
        self.h2 = Dense(32, activation=relu)
        self.h3 = Dense(16, activation=relu)
        self.mu = Dense(action_dim, activation=tanh)
        self.std = Dense(action_dim, activation=softplus)
        # self.mu = Lambda(lambda x: x * self.action_bound) # 책에선 굳이 Lambda 층을 초기화 함수에서 선언하지 않는다.
        # 파라미터가 없기 때문인듯

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        mu = self.mu(x)
        mu = Lambda(lambda x: x*self.action_bound)(mu)
        # -> (-1, 1) -> (-2, 2) 범위 조정
        std = self.std(x)

        return [mu, std]

class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.h1 = Dense(64, activation=relu)
        self.h2 = Dense(32, activation=relu)
        self.h3 = Dense(16, activation=relu)
        self.v = Dense(1, activation='linear') # linear안써도 될거같은데... todo

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        v = self.v(x)

        return v

## PPO 에이전트 클래스
class PPOagent(object):
    def __init__(self, env: gym.Env, args: argparse.Namespace):
        # IDE 코딩에도 편하고 CLI 에서도 편하게 사용하기 위한 argument hadling..
        self.GAMMA = args.gamma
        self.GAE_LAMBDA = args.gae_lambda
        self.BATCH_SIZE = args.batch_size
        self.ACTOR_LEARNING_RATE = args.actor_learning_rate
        self.CRITIC_LEARNING_RATE = args.critic_learning_rate
        self.RATIO_CLIPPING = args.ratic_clipping
        # ...

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0] #  action은 1차원 연속값
        # 표준편차의 최솟값과 최댓값 설정
        self.std_bound = [0.01, 1.0] # std가 0이 되면 결정적이 되므로 탐색을 안하게 될 거니깐 하한선도 정해줘야함. 상한선도 안 정해주면 터무니 없는
        # 값으로 커질 수가 있는데 이때는 너무 탐색을 많이 하게 돼서 수렴하는데 너무 오래 걸리게 됨.

        # 액터 신경앙 및 크리틱 신경망 생성
        self.actor = Actor(self.action_dim, self.action_bound)
        self.critic = Critic()
        # self.actor.build(input_shape=(None, self.state_dim)) todo build 해야하는지 겁증할 것
        # self.critic.build(input_shape=(None, self.state_dim)) todo

        self.actor.summary()
        self.critic.summary()

        # Optimizer
        self.actor.opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic.opt = Adam(self.CRITIC_LEARNING_RATE)

        # 에피소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = []

    ## 로그-정책 확률밀도함수 계산
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5 * (action - mu)**2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    ## 액터 신경망으로 정책의 평균, 표준편차를 계산하고 행동 샘플링
    def get_policy_action(self, state):
        mu_a, std_a = self.actor(state)
        mu_a = mu_a.numpy()[0] # .. 첫번째 행의 액션만 선택.. 일종의 squeezing... + parameter-freezing 효과도 있을 듯
        std_a = std_a.numpy()[0]
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])

        action = np.random.normal(mu_a, std_a, size=self.action_dim)
        return mu_a, std_a, action

    ## GAE와 시간차 타깃 계산
    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros.like(rewards)
        gae = np.zerors.like(rewards)
        gae_cumulative = 0
        forward_val = 0 # done일 경우 next_v_value는 논리상 0이 되어야 함.

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.GAMMA * forward_val - v_values[k]
            gae_cumulative = self.GAMMA * self.GAE_LAMBDA * gae_cumulative + delta
            gae[k] = gae_cumulative
            n_step_targets[k] = gae[k] + v_values[k]
            forward_val = v_values[k]

        return gae, n_step_targets

    ## 배치에 저장된 데이터 추출 #
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)

        return unpack # 아니 이거 그냥 squeeze 하는거 같은데 굳이 이런식으로 해야함??
rewards1 = [[1],[2],[3],[4]]
unpack_batch(rewards1)
rewards1.shape
rewards1.reshape()

if __name__ == "__main__":
    args = get_args()
    print(f"args.GAMMA: {args.gamma}, args.GAE_LAMBDA: {args.gae_lambda}")
    print(args)


