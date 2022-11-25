import pygame
import random
import numpy as np
from utils import *
pygame.init()

GRID_SIZE = 40
BLOCK_SIZE = 20
SPEED = 0

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
        pygame.display.set_caption('Tron AI')
        self.clock = pygame.time.Clock()

        # Number of games
        self.nb_games = 0

        self.reset()
    
    def reset(self):
        # Grid State Init
        self.grid_state = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid_state[1:-1, 1:-1] = np.zeros((self.grid_size-2, self.grid_size-2), dtype=np.int32)
        self.grid_state_obstacle = self.grid_state.copy()

        # Players Init
        self.num_player = len(self.players)

        # Init Game State, Player starting position and direction
        for player in self.players:
            pos_init = np.array([random.randint(1, self.grid_size-1), random.randint(1, self.grid_size-1)])

            while self.grid_state[pos_init[0], pos_init[1]] != 0:
                pos_init = np.array([random.randint(1, self.grid_size-1), random.randint(1, self.grid_size-1)])

            player.init_pos(pos_init)
            self.grid_state[pos_init[0], pos_init[1]] = player.num+2
            self.grid_state_obstacle[pos_init[0], pos_init[1]] = 1
            player.set_dir(random.choice(list(Direction)))
        
        self.game_over = False
        
    def play_step(self, player):
        player.lifetime += 1

        # Reward
        # reward = (len(self.players) - self.num_player + 1) * np.log(max(player.lifetime/4, 1))
        reward = len(self.players) - self.num_player

        # Quit the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move the player 1 cell in its direction
        player.move()

        # Check collision
        if self.is_colliding(player):
            # reward = -np.exp(1/(reward+1))*(self.num_player+1)**2
            reward = -player.lifetime

        # Update the grid
        if not player.dead:
            self.grid_state[player.get_pos()[0], player.get_pos()[1]] = player.num+2
            self.grid_state_obstacle[player.get_pos()[0], player.get_pos()[1]] = 1

        # Update UI and Clock
        self.update_ui()
        self.clock.tick(SPEED)

        # Return if game is over (0 player alive)
        if self.num_player <= 0: self.game_over = True

        return reward
    
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