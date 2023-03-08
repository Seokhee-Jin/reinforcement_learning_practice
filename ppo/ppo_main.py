# PPO main
# coded by St.Watermelon
## PPO 에이전트를 학습하고 결과를 도시하는 파일

from ppo.ppo_learn import PPOagent
import gymnasium as gym

"""
import tensorflow as tf
env.reset()
state , _, _, _, _ = env.step([0])
tf.convert_to_tensor([state], dtype=tf.float32)
"""
def main():
    max_episode_num = 1000
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = PPOagent(env)

    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()

if __name__ == "__main__":
    main()