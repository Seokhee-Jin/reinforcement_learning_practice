import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        ###
        i = 0
        ###
        for r, prob in self.data[::-1]: # 아하 모든 상태가 start가 될 수 있다고 가정하고
            R = r + gamma * R
            loss = -R * torch.log(prob)
            loss.backward() # for 문을 돌면서 loss 값이 누적되며 더해짐..

        self.optimizer.step()
        self.data = []
