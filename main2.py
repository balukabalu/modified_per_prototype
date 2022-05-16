# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym

from ddpg_for_per import DDPG


from replaymemory import ReplayMemory
from prioritized_memory import Memory
from prioritized_memory_2 import Memory2

CLASSIC_DDPG = 0
PER_ORIGINAL = 1
PER_MODIFIED = 2



def main(mode=0):
    LRA = 0.01
    LRC = 0.001
    GAMMA = 0.99
    TAU = 0.01
    BUFFER_SIZE = 256
    BATCH_SIZE = 64
    HIDDEN_SIZE = 128

    EPISODES = 10000

    epsilon = 1

    env = gym.make("MountainCarContinuous-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = DDPG(state_size, action_size, HIDDEN_SIZE, TAU, LRA, LRC, GAMMA, BATCH_SIZE)
    if mode == CLASSIC_DDPG:
        memory = ReplayMemory(BUFFER_SIZE)
    if mode == PER_ORIGINAL:
        memory = Memory(BUFFER_SIZE)
    if mode == PER_MODIFIED:
        memory = Memory2(BUFFER_SIZE)


    for i in range(EPISODES):



        done = False
        score = 0

        s_t = env.reset()
        s_t = np.reshape(s_t, [1, state_size])

        j = 0

        while True:
            if True:
                env.render()

            action = agent.get_action(s_t)

            print(action)



            s_t1, reward, done, info = env.step(action)
            s_t1 = np.reshape(s_t1, [1, state_size])

            reward = reward if not done or score == 499 else -10

            #ide jön a TD error
            if mode == CLASSIC_DDPG:
                memory.add(s_t, action, reward, s_t1, done)
                minibatch = memory.getBatch(BATCH_SIZE)
            if mode == PER_ORIGINAL:
                R = agent.calculate_error_for_per(s_t, action, reward, s_t1, done)
                memory.add(R, (s_t, action, reward, s_t1, done))
                minibatch, idxs, is_weight = memory.sample(BATCH_SIZE)
            if mode == PER_MODIFIED:
                R = agent.calculate_error_for_per(s_t, action, reward, s_t1, done)
                memory.add(R, (s_t, action, reward, s_t1, done))
                minibatch, idxs, is_weight, times_sel = memory.sample(BATCH_SIZE)
                times_sel = np.array(times_sel)

            if j > agent.train_start:

                if mode == CLASSIC_DDPG:
                    agent.train_model(minibatch)
                if mode == PER_ORIGINAL:

                    errors = agent.train_model_per(minibatch)


                    for i in range(BATCH_SIZE):
                        idx = idxs[i]
                        memory.update(idx, errors[i])

                if mode == PER_MODIFIED:

                    times_sel = times_sel.reshape(times_sel.shape[0],-1)

                    errors = agent.train_model_per(minibatch)
                    ucts = errors + memory.cp * np.sqrt(np.log(times_sel)/times_sel.max())

                    for i in range(BATCH_SIZE):
                        idx = idxs[i]
                        memory.update(idx, ucts[i])


            score += reward
            s_t = s_t1

            j+=1

            if j%100 == 0:
                print(j)

            if done:
                break



        print( i, " -th Episode done, steps:  " , j, "    Score: ", score)








if __name__ == '__main__':

    main(CLASSIC_DDPG)

