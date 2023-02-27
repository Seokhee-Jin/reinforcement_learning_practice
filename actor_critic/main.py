from actor_critic import *
import gym
from torch.distributions import Categorical

n_rollout = 10
def main():
    env = gym.make("CartPole-v1")
    model = ActorCritic()
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        done = False
        s, _ = env.reset()

        while not done:
            for t in range(n_rollout): # episode가 끝날때까지 기다리진 않지만 그래도 10번동안은 그냥 기다린다.. episode보다 작은 단위일 뿐.
                prob = model.pi(torch.from_numpy(s).float()) # 기본적으로 신경망에는 인풋을 float()으로 해야하는듯. float들과 곱해질 예정이니.
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, _, _  = env.step(a)
                model.put_data((s, a, r, s_prime, done))

                s = s_prime
                score += r

                if done:
                    break

            model.train_net()

        if n_epi%print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()
