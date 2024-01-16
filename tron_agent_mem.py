import random
import numpy as np
import torch
import torch.nn.functional as F
from utils import *
from collections import deque
from model import DQN, DQTrainer

MAX_GAME = 75
MAX_MEMORY = 100_000
STATE_MEMORY = 5
BATCH_SIZE = 1000
LR = 0.001

class TronPlayer:
    def __init__(
        self, 
        num :int, 
        color :tuple,
        radius :int = 5, 
        epsilon :float = 0.8, 
        gamma :float = 0.9
        ) -> None:

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
        self.state_memory = deque([torch.zeros((2*self.radius + 1, 2*self.radius + 1)) for _ in range(STATE_MEMORY)], maxlen=STATE_MEMORY)
        self.model = DQN(len(Action), STATE_MEMORY, 32).to(self.device)
        self.trainer = DQTrainer(self.model, lr=LR, gamma=self.gamma, device=self.device)

        self.reset()
    
    # Reset the board and modify the exploration/exploitation ratio
    def reset(self) -> None:
        self.dead = False
        self.lifetime = 0
        if self.epsilon <= 0.05:
            self.epsilon = 0
        self.epsilon *= (1 - 1/(0.2*MAX_GAME))
    
    # Load a model checkpoint from a .pth file
    def load_from_save(self, state_dict_path='./model/model.pth') -> None:
        self.nb_games = 0
        self.model.load_state_dict(torch.load(state_dict_path))
        self.trainer = DQTrainer(self.model, lr=LR, gamma=self.gamma, device=self.device)

    def init_pos(self, pos) -> None:
        self.pos = pos
    
    def set_dir(self, dir) -> None:
        self.direction = dir
    
    # Update Direction depending on the choosen action (Maybe better way to do it)
    def update_dir(self, action) -> None:
        idx = DIR_LIST_CW.index(self.direction)

        if action == Action.RIGHT:
            self.direction = DIR_LIST_CW[(idx+1)%4]
        elif action == Action.LEFT:
            self.direction = DIR_LIST_CW[(idx-1)%4]
    
    def move(self) -> None:
        self.pos += np.array(self.direction.value)
    
    def is_dead(self) -> bool:
        return self.dead
    
    def get_color(self) -> tuple:
        return self.color
    
    def get_pos(self) -> np.ndarray:
        return self.pos
    
    # Retrieve a patch of the play area for training
    def get_vision(self, grid) -> torch.Tensor:
        vision = grid.copy()
        vision[self.pos[0], self.pos[1]] = 10 * (DIR_LIST_CW.index(self.direction) + 2)
        vision = np.pad(vision, pad_width=self.radius, constant_values=1)
        vision = vision[self.pos[0]:self.pos[0]+2*self.radius+1, self.pos[1]:self.pos[1]+2*self.radius+1]
        # return torch.tensor(np.rot90(vision, DIR_LIST_CW.index(self.direction)).copy(), dtype=torch.float)/2
        # return torch.tensor(np.rot90(vision, DIR_LIST_CW.index(self.direction)).copy(), dtype=torch.float)
        return torch.tensor(vision.copy(), dtype=torch.float)
    
    # Save a game in the memory (current state, reward given, action taken, resulting state, and if the agent is dead)
    # and stack it with some previous game (game buffer)
    def save2mem(self, reward, action, next_state) -> None:
        cur_state = self.state_memory
        self.state_memory.appendleft(next_state)
        self.memory.append((torch.stack(tuple(cur_state)), reward, torch.tensor(action, dtype=torch.long), torch.stack(tuple(self.state_memory)), self.dead))

    # Long term memory training by taking BATCH_SIZE games from the memory
    # and training the network on them.
    def train_longmem(self) -> None:
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        cur_state, reward, action, next_state, dead = zip(*mini_sample)
        self.trainer.train_step(torch.stack(cur_state), torch.tensor(reward, dtype=torch.float), torch.stack(action), torch.stack(next_state), torch.tensor(dead))

    # Short term momory training by taking the last game and training the network on it
    # and stack the last game with some previous game (game buffer)
    def train_shortmem(self, reward, action, new_state) -> None:
        cur_state = self.state_memory
        self.state_memory.appendleft(new_state)
        self.trainer.train_step(torch.stack(tuple(cur_state)), torch.tensor(reward, dtype=torch.float), torch.tensor(action, dtype=torch.long), torch.stack(tuple(self.state_memory)), self.dead)

    # Get an action from the network or by randomness
    def get_action(self, state) -> Action:
        # random  moves (exploration/exploitation)
        if random.random() < self.epsilon:
            action = random.choice(ACTION_LIST)
        else:
            self.state_memory.appendleft(state)
            state0 = torch.stack(tuple(self.state_memory)).clone().detach().unsqueeze(dim=0)
            pred = self.model(state0.to(self.device))
            action = ACTION_LIST[torch.argmax(pred).item()]
        return action