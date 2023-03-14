# DDPG main (tf2 subclassing API version)
# coded by St.Watermelon

import gymnasium as gym
from ddpg_learn import DDPGagent

def main():

    max_episode_num = 200
    env = gym.make("Pendulum-v1")
    agent = DDPGagent(env)

    agent.train(max_episode_num)

    agent.plot_result()



if __name__=="__main__":
    main()