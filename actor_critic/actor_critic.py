import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

learning_rate = 0.0002
gamma = 0.98

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim) #softmax_dim ?
        return prob

    def v(self, x): # forward가 아니므로 직접 이름 pi, v를 사용해야 호출되겠구나
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s) # s는 리스트이므로 대괄호 한번 더 씌울 필요가 없음
            a_lst.append([a])
            r_lst.append([r/100.0]) #... 왜 100으로 나눠주는 걸까. 보상을 정규화? 입력이되는 obs랑 rew정규화는 필수일까?
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), \
                                                               torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), \
                                                               torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float),
        self.data = [] # 데이터 지우고 반환.. pop처럼
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a) #아니 이게 뭐야. 이미 에피소드가 끝난 상태에서 하는거임(-> 에피가 끝난건 아니고 10번정도 진행한걸 모아놈..)?? 배치학습 하려면 어쩔 수 없는건가..
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach()) #각가 pi하고 v빼고는 전부 상수취급.
        # critic에서 target을 상수취급하는 이유는 DQN에서 별도의 네트워크를 상수취급하여 쓰는 이유과 같은 이유. 타겟이 변하지 않는 것이 학습안정성을 높임.

        # smooth l1 loss는 기본적으로 예측값과 관측값의 차이의 절대값을 의미하는 li loss에서 미분이 불가한 지점인 0 주변은 l2(mse)로 계산하도록하는..
        # 즉 0주변을 부드럽게 한 l1 loss로서, 이상치에 덜민감한 l1 loss와 미분이 가능한 l2 loss의 장점을 합친 loss 이다.. 흠.
        self.optimizer.zero_grad()
        loss.mean().backward() # 배치를 평균해서 한번에 역전파..
        self.optimizer.step()






