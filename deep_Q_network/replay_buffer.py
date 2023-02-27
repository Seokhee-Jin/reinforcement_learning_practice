import collections
import numpy as np
import torch

matrix = torch.range(1,8).reshape(2,2,2)
indices = torch.tensor([0,1,0,1]).reshape(2,2)
indices = indices.unsqueeze(-1)
indices
matrix.gather(1, indices)

matrix = torch.range(1,1000).view()

import random

buffer_limit = 50000


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        #mini_batch = np.random.choice(self.buffer, n)
        np.random.
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        #print("a_lst: {}, r_lst: {}, s_prime_lst: {}".format(a_lst, r_lst, s_prime_lst))
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(
            s_prime_lst, dtype=torch.float), torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)
