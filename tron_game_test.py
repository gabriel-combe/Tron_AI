import pygame
import random
import numpy as np
from enum import Enum

pygame.init()

class Direction(Enum):
    UP = [-1,0]
    RIGHT = [0,1]
    DOWN = [1,0]
    LEFT = [0,-1]

class Action(Enum):
    FORWARD = [1, 0, 0]
    RIGHT = [0, 1, 0]
    LEFT = [0, 0, 1]

DIR_LIST_CW = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

WHITE = (255,255,255)
GRAY = (50,50,50)
BLACK = (0,0,0)

GRID_SIZE = 40
BLOCK_SIZE = 20
SPEED = 40

class TronGame:
    def __init__(self, players, grid_size=GRID_SIZE, block_size=BLOCK_SIZE):
        # Display Settings
        self.grid_size = grid_size
        self.block_size = block_size
        self.size = self.grid_size*self.block_size

        # Players Init
        self.players = players
        
        # Init display
        self.display = pygame.display.set_mode((self.size, self.size))
        pygame.display.set_caption('Tron')
        self.clock = pygame.time.Clock()

        self.reset()
        
        
    
    def reset(self):
        # Grid State Init
        self.grid_state = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid_state[1:-1, 1:-1] = np.zeros((self.grid_size-2, self.grid_size-2), dtype=np.int32)

        # Players Init
        self.num_player = len(self.players)

        # Init Game State, Player starting position and direction
        for player in self.players:
            pos_init = np.array([random.randint(1, self.grid_size-1), random.randint(1, self.grid_size-1)])

            while self.grid_state[pos_init[0], pos_init[1]] != 0:
                pos_init = np.array([random.randint(1, self.grid_size-1), random.randint(1, self.grid_size-1)])

            player.init_pos(pos_init)
            self.grid_state[pos_init[0], pos_init[1]] = player.num+2
            player.set_dir(random.choice(list(Direction)))
        
        self.frame_iteration = 0


    def play_step(self):
        self.frame_iteration += 1

        # Quit the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Update Players position
        for player in self.players:

            # Check if player is already Dead
            if player.is_dead(): continue

            # Randomly choose an action
            if random.random() > 0.6: player.update_dir(random.choice(list(Action)))

            # Move the player 1 cell in its direction
            player.move()

            # Check collision
            if self.is_colliding(player): continue

            # Update the grid
            self.grid_state[player.get_pos()[0], player.get_pos()[1]] = player.num+2

        # Update UI and Clock
        self.update_ui()
        self.clock.tick(SPEED)

        # Return if game is over (1 player alive)
        if self.num_player <= 1:
            return True

        return False
    
    def update_ui(self):
        # Background color
        self.display.fill(BLACK)

        # Display every cell of the grid with the corresponding color
        for i in range(self.grid_state.shape[0]):
            for j in range(self.grid_state.shape[1]):
                # 0 if the cell is Empty
                if self.grid_state[i,j]==0: 
                    pygame.draw.rect(self.display, WHITE, pygame.Rect((j*self.block_size)+1, (i*self.block_size)+1, self.block_size-1, self.block_size-1))
                # 1 if the cell is a Wall
                elif self.grid_state[i,j]==1:
                    pygame.draw.rect(self.display, GRAY, pygame.Rect((j*self.block_size)+1, (i*self.block_size)+1, self.block_size-1, self.block_size-1))
                # All the remaining number correspond to a player or its trail
                else:
                    pygame.draw.rect(self.display, self.players[self.grid_state[i,j]-2].get_color(), pygame.Rect((j*self.block_size)+1, (i*self.block_size)+1, self.block_size-1, self.block_size-1))

        pygame.display.flip()

    def is_colliding(self, player):
        # Check collision
        if self.grid_state[player.get_pos()[0],player.get_pos()[1]] != 0:
            player.dead = True
            self.num_player -= 1
            return True
        return False

class TronPlayer:
    def __init__(self, num, color, radius=5):
        self.num = num
        self.color = color
        self.radius = radius
        self.pos = np.array([0,0])
        self.direction = Direction.UP
        self.dead = False
    
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
        padded_grid = np.pad(grid, pad_width=self.radius, constant_values=1)
        return padded_grid[self.pos[0]:self.pos[0]+2*self.radius+1, self.pos[1]:self.pos[1]+2*self.radius+1]

if __name__=='__main__':
    # Create a game with 4 players
    game = TronGame([TronPlayer(0, (0,0,255)), TronPlayer(1, (0,255,0)), TronPlayer(2, (255,0,0)), TronPlayer(3, (255,255,0))])

    # Game Loop
    while True:
        game_over = game.play_step()

        # Break if Game Over
        # if game_over: break
    
    # Quit Pygame
    pygame.quit()