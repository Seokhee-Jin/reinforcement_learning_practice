# 출처: 수학으로 풀어보는 강화학습 원리와 알고리즘
# A3C learn (tf2 subclassing API version: Data)
# coded by St.Watermelon

import gymnasium as gym
import keras.activations
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Lambda
from tensorflow.python.keras.optimizer_v2.adam import Adam # 언제 바뀔 지 모르므로 하이퍼파라미터 조정하지 않는 이상 'adam'쓰는게 나을 듯.

import numpy as np
import matplotlib.pyplot as plt

import threading
import multiprocessing

# 액터 신경망
class Actor(Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_dim

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.mu = Dense(action_dim, activation=tf.keras.activations.tanh)  # -1,1로 반환하는 비선형 활성화함수 => tanh
        self.lamb = Lambda(lambda x: x * self.action_bound)  # 평균값을 [-action_bound, action_bound] 범위로 조정
        self.std = Dense(action_dim, activation=tf.keras.activations.softplus)  # 0,inf로 반환하는 비선형 활성화함수 => softplus

    def call(self, state):  # inputs로 state를 받는 네트워크 모델
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        mu = self.mu(x)
        mu = self.lamb(mu)
        std = self.std(x)

        return [mu, std]

        '''# 평균값을 [-action_bound, action_bound] 범위로 조정
        mu = Lambda(lambda x: x*self.action_bound)(mu)''' # 흠.. call층에서 한 이유가 있을까?

## 크리틱 신경망

class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.v = Dense(1, activation='linear')  # 'linear'안쓰고 그냥 None으로 냅둬도 똑같을텐뎅

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        v = self.v(x)

        return v

# 모든 워커에서 공통으로 사용할 글로벌 변수 설정
global_episode_count = 0
global_step = 0
global_episode_reward = []

## a3c 에이전트 클래스
class A3Cagent(object): # 굳이 object를 넣어주는 이유: 명확성 혹은 python2와의 호환성을 위함일 수 있지만.. 보통 취향 차이라고 함.
    def __init__(self, env_name):

        # 학습할 환경 설정
        self.env_name = env_name
        self.WORKER_NUM = multiprocessing.cpu_count() # 내 컴퓨터 cpu가 6코어 12쓰레드라서 12가 반환됨.
        env = gym.make(env_name)
        # 상태변수 차원
        self.state_dim




