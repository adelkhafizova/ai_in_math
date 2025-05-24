import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import sys
sys.path.append('../../scripts')
import burau_enchanced as b
from free_scripts import largest_power_range_word

MODULO = 2
MAX_LENGTH = 32
matrices= {
    "A":b.A.convert_to_modulo(MODULO),
    "B": b.B.convert_to_modulo(MODULO),
    "a": b.a.convert_to_modulo(MODULO),
    "b": b.b.convert_to_modulo(MODULO)
}


   
class BurauEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.turn = 0
        self.word = np.zeros(MAX_LENGTH, dtype=int)  # initialized with all zeroes
        self.winner = None
        self.done = False
        return self.word.copy()
    
    def step(self, action):
        if self.done:
            raise ValueError("Game has ended. Call reset().")


        # Apply move
        self.word[self.turn] = action
        self.turn += 1
        # Check for game over
        
        self.done, self.winner = self.check_game_over()
        reward = self.get_reward_test()
        
        return self.word.copy(), reward, self.done

    def get_reward(self):
        if self.turn != 1 and (self.word[self.turn-1] + self.word[self.turn-2] == 5):
            return -10000
        range = largest_power_range_word(matrices,self.render())
        return 10/(1+range)
    
    def get_reward_test(self):
        if self.turn != 1 and (self.word[self.turn-1] + self.word[self.turn-2] == 5):
            return -10
        word = self.render()
        if len(word) != 1:
            range_before = largest_power_range_word(matrices,word[:-1])
            range_after = largest_power_range_word(matrices,word)           
            diff =  range_before - range_after
            return diff
        else:
            return 0
    def check_game_over(self):
        if self.turn == MAX_LENGTH:
            return True, 0
        return False, None

    def render(self):
        symbols = {0:"",1:"A",2:"B",3:"b",4:"a"}
        word_repr = ""
        for i in self.word:
            word_repr += symbols[i]
        return word_repr


class DQNBurau(nn.Module):
    def __init__(self):
        super(DQNBurau, self).__init__()
        self.fc1 = nn.Linear(MAX_LENGTH, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)  # 9 actions (board positions)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # make batch of 1
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # raw Q-values for each action
    

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)
    

