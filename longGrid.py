import numpy as np
from env import Env

class State(object):
    def __init__(self, index, reward, isTerminal = False):
        self.index = index
        self.reward = reward
        self.isTerminal = isTerminal

    def isTerminal(self):
        return self.isTerminal

    def getReward(self):
        return self.reward

    def getIndex(self):
        return self.index


class LongGrid(Env):

    def __init__(self, width, reward):
        self.stateList = list()
        for i in range(width - 1):
            state = State(i , 0)
            self.stateList.append(state)
        # Terminal state
        self.stateList.append(State(width, reward, True))

    def getNextStateWithAction(self, state, action):
        index = state.getIndex()
        if action == 'left':
            index = index - 1
        elif action == 'right':
            index = index + 1 
        if index < 1 or index > self.width:
            index = state.getIndex()
        return self.stateList[index]

    def getPossibleActions(self, state):
        actions = list()
        if state.getIndex == 1: # start state
            actions.append('right')
        elif not state.isTerminal: # not terminal state
            actions.append('left')
            actions.append('right')
        return actions

    def getRestartState(self):
        index = np.random.random_integers(self.width)
        return self.stateList[index]

    def getReward(self, state, action, nextState):
        return nextState.getReward()

    def isTerminal(self, state):
        return state.isTerminal()

    def state2feature(self, state):
        return state.getIndex()

    def action2feature(self, action):
        if action == 'left':
            return -1
        elif action == 'right':
            return 1
