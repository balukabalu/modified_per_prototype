import random
import numpy as np
from SumTree2 import SumTree

import torch
from torch.autograd import Variable


class Memory2:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    beta = 0.6
    alpha = 0.4
    alpha_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.cp = 1

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.beta

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self,n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        times_sel = []


        self.alpha = np.min([1., self.alpha + self.alpha_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data, ts) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
            times_sel.append(ts)


        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities , -self.alpha)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight, times_sel

    def update(self, idx, error):
        p = self._get_priority(error)

        self.tree.update(idx, p)


