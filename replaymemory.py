from collections import deque
import random
import numpy as np


class ReplayMemory():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):

        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(list(self.buffer), batch_size)

    def getBatch2(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return self.buffer[np.random.choice(self.buffer.shape[0], size = self.num_experiences, replace = False),:]
        else:
            self.buffer.shape[0]
            np.random.choice(self.buffer.shape[0], size=batch_size, replace=False)
            #print(self.buffer.size)
            return self.buffer[np.random.choice(self.buffer.shape[0], size = batch_size, replace = False),:]

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):

        experience = (state, action, reward, new_state, done)

        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def add2(self, state, action, reward, new_state, done):

        experience = np.asarray([[state, action, reward, new_state, done]])

        if self.num_experiences < self.buffer_size:
            self.buffer = np.append(self.buffer, experience, axis=0)
            self.num_experiences += 1

        else:
            print("itt vagyok!")
            print(self.buffer[0])
            self.buffer = np.delete(self.buffer, 0, axis=0)
            print(self.buffer.size)
            self.buffer = np.append(self.buffer, experience)
            print(self.buffer.size)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = np.empty((0, 5))
        self.num_experiences = 0
