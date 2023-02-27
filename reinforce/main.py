import gym
from policy import *
from torch.distributions import Categorical

def main():
    env = gym.make('CartPole-v1')
    pi = Policy() # 4개의 특징을 가진 인풋을 넣으면 두 행동의 확률을 반환하는 정책 네트워크
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False

        while not done:
            prop = pi(torch.from_numpy(s).float())
            m = Categorical(prop)
            a = m.sample()
            s_prime, r, done, _, _ = env.step(a.item())
            pi.put_data((r, prop[a]))
            s = s_prime
            score += r

        pi.train_net()
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode : {}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0

        env.close()

if __name__ == "__main__":
    main()

