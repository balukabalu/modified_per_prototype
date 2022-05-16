import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable

from actormodel import Actor
from criticmodel import Critic
from replaymemory import ReplayMemory
from ou import OU
#from util import *

ou = OU()
criterion = nn.MSELoss()

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)



class DDPG():
    def __init__(self, state_size, action_size, hidden_size, tau, lra, lrc, gamma, batch_size):

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.tau = tau
        self.lra = lra
        self.lrc = lrc
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1

        self.actor = Actor(state_size, action_size, hidden_size)
        self.critic = Critic(state_size, action_size, hidden_size)
        self.actor_target = Actor(state_size, action_size, hidden_size)
        self.critic_target = Critic(state_size, action_size, hidden_size)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lra)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = lrc)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        #self.replay_memory = ReplayMemory(10000)
        self.ou = OU()
        self.is_training = 1

        self.train_start = 120


    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )


    def get_action(self, s_t, decay_epsilon = True):
        action = np.array(self.actor(torch.from_numpy(np.array([s_t])).float()).detach()).squeeze(0)
        #action = self.actor(torch.from_numpy(np.array([s_t])).float())


        action += self.is_training * max(self.epsilon, 0)* self.ou.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= 0.0001

        return action

    def train_model(self, batch):

        """#print(batch)
        states = np.asarray([e[0] for e in batch]).reshape(64,-1)
        #print(states.shape)
        states2 = states.reshape(64,-1)
        #print(states2.shape)
        #print("statesek voltak ezek")
        actions = np.asarray([e[1] for e in batch]).reshape(64,-1)
        rewards = np.asarray([e[2] for e in batch]).reshape(64,-1)
        y_t = np.asarray([e[1] for e in batch]).reshape(64,-1)
        next_states = np.asarray([e[3] for e in batch]).reshape(64,-1)
        dones = np.asarray([e[4] for e in batch]).reshape(64,-1)"""

        #for e in batch:

        states = np.asarray([e[0] for e in batch]).reshape(64,-1)
        actions = np.asarray([e[1] for e in batch]).reshape(64,-1)
        rewards = np.asarray([e[2] for e in batch]).reshape(64,-1)
        next_states = np.asarray([e[3] for e in batch]).reshape(64,-1)

        dones = np.asarray([e[4] for e in batch]).reshape(64,-1)
        y_t = np.asarray([e[1] for e in batch]).reshape(64,-1)

        #print("next")
        #print(next_states[0])

        target_q_values = self.critic_target([torch.from_numpy(next_states), self.actor_target(torch.from_numpy(next_states))])

        for k in range(len(rewards)):

                """print(type(torch.from_numpy(np.asarray([self.gamma]))))
                print(type(torch.from_numpy(dones.astype(np.float))))
                print(type(target_q_values))
                #y_t[k] =  torch.from_numpy(np.asarray([self.gamma])) *  torch.from_numpy(dones.astype(np.float)) * target_q_values
                print
                np.asarray([self.gamma]) * dones.astype(np.float) * target_q_values.detach().numpy()"""
                y_t =  np.asarray(y_t) + np.asarray([self.gamma]) *  dones.astype(np.float) * target_q_values.detach().numpy()*dones.astype(np.float)

           # to_tensor(reward_batch) + \
            #self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values

        self.critic.zero_grad()

        q_batch = torch.Tensor(self.critic([torch.from_numpy(states), torch.from_numpy(actions)]))
        # PER PRED
        """print(q_batch.size())
        print(type(y_t))
        print(q_batch.dtype)"""
        y_t = torch.from_numpy(y_t.astype(np.float))    #PER TARGET

        #print(y_t.dtype)
        #print("asd")




        value_loss = criterion(q_batch.float(), y_t.float())

        #print(value_loss.dtype)

        value_loss.backward()
        self.critic_optimizer.step()

        self.actor.zero_grad()

        policy_loss = -self.critic([torch.from_numpy(states), self.actor(torch.from_numpy(states))])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def train_model_per(self, batch):

        """#print(batch)
        states = np.asarray([e[0] for e in batch]).reshape(64,-1)
        #print(states.shape)
        states2 = states.reshape(64,-1)
        #print(states2.shape)
        #print("statesek voltak ezek")
        actions = np.asarray([e[1] for e in batch]).reshape(64,-1)
        rewards = np.asarray([e[2] for e in batch]).reshape(64,-1)
        y_t = np.asarray([e[1] for e in batch]).reshape(64,-1)
        next_states = np.asarray([e[3] for e in batch]).reshape(64,-1)
        dones = np.asarray([e[4] for e in batch]).reshape(64,-1)"""

        # for e in batch:

        states = np.asarray([e[0] for e in batch]).reshape(64, -1)
        actions = np.asarray([e[1] for e in batch]).reshape(64, -1)
        rewards = np.asarray([e[2] for e in batch]).reshape(64, -1)
        next_states = np.asarray([e[3] for e in batch]).reshape(64, -1)

        dones = np.asarray([e[4] for e in batch]).reshape(64, -1)
        y_t = np.asarray([e[1] for e in batch]).reshape(64, -1)

        print(states)
        # print("next")
        # print(next_states[0])

        target_q_values = self.critic_target(
            [torch.from_numpy(next_states), self.actor_target(torch.from_numpy(next_states))])

        for k in range(len(rewards)):
            """print(type(torch.from_numpy(np.asarray([self.gamma]))))
            print(type(torch.from_numpy(dones.astype(np.float))))
            print(type(target_q_values))
            #y_t[k] =  torch.from_numpy(np.asarray([self.gamma])) *  torch.from_numpy(dones.astype(np.float)) * target_q_values
            print
            np.asarray([self.gamma]) * dones.astype(np.float) * target_q_values.detach().numpy()"""
            y_t = np.asarray(y_t) + np.asarray([self.gamma]) * dones.astype(
                np.float) * target_q_values.detach().numpy() * dones.astype(np.float)

        # to_tensor(reward_batch) + \
        # self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values

        self.critic.zero_grad()

        q_batch = torch.Tensor(self.critic([torch.from_numpy(states), torch.from_numpy(actions)]))
        # PER PRED
        """print(q_batch.size())
        print(type(y_t))
        print(q_batch.dtype)"""
        y_t = torch.from_numpy(y_t.astype(np.float))  # PER TARGET

        # print(y_t.dtype)
        # print("asd")

        errors = torch.abs(q_batch - y_t).data.numpy()

        value_loss = criterion(q_batch.float(), y_t.float())

        # print(value_loss.dtype)

        value_loss.backward()
        self.critic_optimizer.step()

        self.actor.zero_grad()

        policy_loss = -self.critic([torch.from_numpy(states), self.actor(torch.from_numpy(states))])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return errors

    def calculate_error_for_per(self, state, action, reward, next_state, done):
        target =torch.Tensor(self.critic([torch.from_numpy(state), torch.from_numpy(action)]))
        old_val = target[0][action]
        #target_val = self.critic_target(Variable(torch.FloatTensor(next_state))).data
        target_val =  self.critic_target([torch.from_numpy(next_state), self.actor_target(torch.from_numpy(next_state))]).data
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.gamma * torch.max(target_val)

        error = abs(old_val - target[0][action])

        return error.detach().numpy()


"""sajt = DDPG(5,1,128,0.99, 0.001, 0.0001, 0.99, 64)

a = [float(0.0),float(1.0),float(2.0),float(3.0),float(4.0)]
b = sajt.get_action(a)

print(b)"""
