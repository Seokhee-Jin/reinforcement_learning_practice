


import gym
from q_net import *
from replay_buffer import *
import torch.optim as optim

learning_rate = 0.0005
gamma = 0.98
batch_size = 32

def train(q:Qnet, q_target:Qnet, memory:ReplayBuffer, optimizer:optim.Optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a) # ?
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) # ?
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
        #Linear annealing from 8% to 1%
        s, _ = env.reset()
        done = False

        while not done:  # 메모리가 2000이상으로 쌓일때까지 랜덤으로 초기화된 q_net으로 행동하면서 s,a,r,s_prime데이터를 쌓는다.
            a = q.sample_action(torch.from_numpy(s).float(), epsilon) # 입실론-그리디로 q_net실행
            s_prime, r, done, _, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r/100.0, s_prime, done_mask))
            s = s_prime
            score += r
            if done:
                break # 이거 없어도 똑같지 않나...

        if memory.size() > 2000:  # 메모리 2000이상 쌓인 순간부터 한 에피소드당 (1)32개의 미니배치 추출 - (2)업데이트 -> 10번 반복하여 q를 훈련시킨다.
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval == 0 and n_epi!=0: # 에피소드 20번마다 q_target 업데이트, 현황 및 스코어(개선정도) 프린트함.
            q_target.load_state_dict(q.state_dict()) #
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
        env.close()

if __name__ == '__main__':
    main()





