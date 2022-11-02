import matplotlib.pyplot as plt
from IPython import display
from enum import Enum

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
ACTION_LIST = [Action.FORWARD, Action.RIGHT, Action.LEFT]

WHITE = (255,255,255)
GRAY = (50,50,50)
BLACK = (0,0,0)

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    for i in range(len(scores)):
        plt.plot(scores[i])
        plt.plot(mean_scores[i])
        plt.text(len(scores[i])-1, scores[i][-1], str(scores[i][-1]))
        plt.text(len(mean_scores[i])-1, mean_scores[i][-1], str(mean_scores[i][-1]))
    plt.ylim(ymin=0)
    plt.show(block=False)
    plt.pause(.1)