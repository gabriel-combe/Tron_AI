import os
from tron_agent_mem import TronPlayer, MAX_GAME
from tron_game import TronGame, GRID_SIZE
from utils import plot
import numpy as np

NUM_AGENT = 4

def train():
    total_score = 0
    records = [0 for _ in range(2)]
    agents = create_agent(NUM_AGENT, GRID_SIZE)
    game = TronGame(agents)

    plot_scores = [ [] for _ in range(len(agents))]
    plot_mean_scores = [ [] for _ in range(len(agents))]

    while True:
        while agents[0].nb_games < MAX_GAME:
            # For all agents
            for agent in agents:
                # If agent is dead skip it
                if agent.dead: continue
                # if agent.num == 0: print(agent.epsilon)

                # get old state
                old_state = agent.get_vision(game.grid_state_obstacle)

                # get move
                final_action = agent.get_action(old_state)

                # Update direction
                agent.update_dir(final_action)
            
            # Then evaluate them
            for agent in agents:
                # If agent is dead skip it
                if agent.dead: continue

                # perform action
                reward = game.play_step(agent)

                # get new state
                new_state = agent.get_vision(game.grid_state_obstacle)

                # Save to Memory
                agent.save2mem(reward, final_action.value, new_state)
                
                # train short memory
                agent.train_shortmem(reward, final_action.value, new_state)

            # Train all agent on their past experience
            if game.game_over:
                # train long memory and plot result
                game.reset()
                game.nb_games += 1

                for agent in agents:
                    agent.nb_games += 1
                    agent.train_longmem()

                    if agent.lifetime > records[0]:
                        records[1] = records[0]
                        records[0] = agent.lifetime
                        if os.path.exists('./model/model_0.pth'):
                            os.replace('./model/model_0.pth', './model/model_1.pth')
                        agent.model.save(f'model_{0}.pth')
                    elif agent.lifetime > records[1]:
                        records[1] = agent.lifetime
                        agent.model.save(f'model_{1}.pth')
                    
                    plot_scores[agent.num].append(agent.lifetime)
                    total_score += agent.lifetime
                    mean_score = total_score/game.nb_games
                    plot_mean_scores[agent.num].append(mean_score)

                    agent.reset()
                
                print('Game ', game.nb_games, 'Records: ', records)

                plot(plot_scores, plot_mean_scores)
        
        for agent in agents:
            agent.epsilon = 0.2
            agent.load_from_save(f'./model/model_{agent.num%(NUM_AGENT//2)}.pth')

def create_agent(num_agent, grid_size):
    return [TronPlayer(i, tuple(np.random.random(size=3) * 256), grid_size) for i in range(num_agent)]

if __name__=='__main__':
    train()