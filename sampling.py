"""
"""

import random
import numpy as np

class Sampling():

    def __init__(self, game, dist, policy, gamma):
        self.game = game
        self.dist = dist
        self.policy = policy
        self.gamma = gamma

    def acceptState(self):
        """
        With prob. self.gamma return true
        """
        return np.random.random() < self.gamma

    def getStartState(self):
        """
        Get a start state from self.dist
        """
        # r = np.random.random()
        # for state, prob in self.dist:
        #     r -= prob
        #     if r <= 0:
        #         return state
        # assert False, "No start state"
        return np.random.choice(self.dict.key(), 1, self.dict.value())

    def getSampledState(self, startState, cutOff):
        """
        Get a sample state from a rollout with prob. self.gamma
        """
        iteration = 1
        currState = startState
        while iteration <= cutOff:
            currState = self.game.getNextState(currState, self.policy[currState])
            if self.acceptState():
                return currState
        return currState

    def sampleStates(self, stateNum, cutOff):
        """
        Sample stateNum states.
        In each rollout, at most cutOff steps will be excuted.
        Return list of states
        """
        sampledStates = []
        for i in range(stateNum):
            startState = self.getStartState()
            sampledState = self.getSampledState(startState, cutOff)
            sampledStates.append(sampledState)

        return sampledStates