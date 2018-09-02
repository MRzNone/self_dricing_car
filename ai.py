# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# creating the architecture of the Neural Network


class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_sizse = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(self.input_sizse, 100)
        self.fc2 = nn.Linear(100, self.nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x

# implementing experience Replay

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.mem = []

    def push(self, event):
        self.mem.append(event)
        if len(self.mem) > self.capacity:
            del self.mem[0]

    def sample(self, batch_size):
        samples = zip(* random.sample(self.mem, batch_size))
        return map(lambda x: Variable(torch.cat(x,0)), samples)

# implementing Deep Q learinig
class Dqn():

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.input_size = input_size
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.mem = ReplayMemory(100000)

        self.optimizer = optim.Adam( self.model.parameters(), lr = 0.001)

        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.ran = 0
        self.load()

    def select_action(self, state):
        guess = self.model(Variable(state, volatile = True))
        probs = F.softmax(guess * 7)  # temperature = 7, higher => higher certainty
        action = probs.multinomial()
        return action.data[0,0]

    def learn(self, batch_state, batch_next_state, batch_action, batch_reward):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = batch_reward + self.gamma * next_outputs
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.mem.push((self.last_state, new_state, torch.LongTensor([int (self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.mem.mem) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.mem.sample(100)
            self.learn(batch_state, batch_next_state, batch_action, batch_reward)

            self.ran += 1
            if self.ran >= 100:
                self.ran = 0
                self.save()
        self.last_action = action
        self.last_reward = reward
        self.last_state = new_state
        self.reward_window.append(reward)
        if len(self.reward_window) > 100:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1)

    def save(self, name = None):
        if name != None:
            torch.save({'state_dic': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                        }, name + str(self.input_size) + '.pth')
        else:
            torch.save({'state_dic': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                        }, 'last_brain' + str(self.input_size) + '.pth')

    def load(self):
        if (os.path.isfile('last_brain' + str(self.input_size) + '.pth')):
            print('=> loading checkpoint...')
            checkpoint = torch.load('last_brain'  + str(self.input_size) + '.pth')
            self.model.load_state_dict(checkpoint['state_dic'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done!')
        else:
            print('No checkpoint found...')
