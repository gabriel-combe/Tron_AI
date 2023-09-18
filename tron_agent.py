import random
import numpy as np
import torch
import torch.nn.functional as F
from utils import *
from collections import deque
from model import DQN, DQTrainer

MAX_GAME = 150
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class TronPlayer:
    def __init__(self, num, color, radius=5, epsilon=0.8, gamma=0.9):
        self.num = num
        self.color = color
        self.radius = radius
        self.pos = np.array([0,0])
        self.direction = random.choice(DIR_LIST_CW)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nb_games = 0

        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN(len(Action), 16).to(self.device)
        self.trainer = DQTrainer(self.model, lr=LR, gamma=self.gamma, device=self.device)

        self.reset()
    
    def reset(self):
        self.dead = False
        self.lifetime = 0
        if self.epsilon <= 0.05:
            self.epsilon = 0
        self.epsilon *= (1 - 1/(0.2*MAX_GAME))
    
    def load_from_save(self, state_dict_path='./model/model.pth'):
        self.nb_games = 0
        self.model.load_state_dict(torch.load(state_dict_path))
        self.trainer = DQTrainer(self.model, lr=LR, gamma=self.gamma, device=self.device)

    def init_pos(self, pos):
        self.pos = pos
    
    def set_dir(self, dir):
        self.direction = dir
    
    # Update Direction depending on the choosen action (Maybe better way to do it)
    def update_dir(self, action):
        idx = DIR_LIST_CW.index(self.direction)

        if action == Action.RIGHT:
            self.direction = DIR_LIST_CW[(idx+1)%4]
        elif action == Action.LEFT:
            self.direction = DIR_LIST_CW[(idx-1)%4]
    
    def move(self):
        self.pos += np.array(self.direction.value)
    
    def is_dead(self):
        return self.dead
    
    def get_color(self):
        return self.color
    
    def get_pos(self):
        return self.pos
    
    def get_vision(self, grid):
        padded_grid = grid.copy()
        padded_grid[self.pos[0], self.pos[1]] = 10 * (DIR_LIST_CW.index(self.direction) + 2)
        # padded_grid = np.pad(padded_grid, pad_width=self.radius, constant_values=1)
        # vision = padded_grid[self.pos[0]:self.pos[0]+2*self.radius+1, self.pos[1]:self.pos[1]+2*self.radius+1]
        # return torch.tensor(np.rot90(vision, DIR_LIST_CW.index(self.direction)).copy(), dtype=torch.float)/2
        # return torch.tensor(np.rot90(vision, DIR_LIST_CW.index(self.direction)).copy(), dtype=torch.float)
        # return torch.tensor(vision.copy(), dtype=torch.float)
        return torch.tensor(padded_grid.copy(), dtype=torch.float)
    
    def save2mem(self, cur_state, reward, action, next_state):
        self.memory.append((cur_state, reward, torch.tensor(action, dtype=torch.long), next_state, self.dead))

    def train_longmem(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        cur_state, reward, action, next_state, dead = zip(*mini_sample)
        self.trainer.train_step(torch.stack(cur_state), torch.tensor(reward, dtype=torch.float), torch.stack(action), torch.stack(next_state), torch.tensor(dead))

    def train_shortmem(self, cur_state, reward, action, next_state):
        self.trainer.train_step(cur_state, torch.tensor(reward, dtype=torch.float), torch.tensor(action, dtype=torch.long), next_state, self.dead)

    def get_action(self, state):
        # random  moves (exploration/exploitation)
        if random.random() < self.epsilon:
            action = random.choice(ACTION_LIST)
        else:
            state0 = state.clone().detach().unsqueeze(dim=0).unsqueeze(dim=0)
            pred = self.model(state0.to(self.device))
            action = ACTION_LIST[torch.argmax(pred).item()]
        return action