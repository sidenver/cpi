"""
Sampling states from a future state distribution
"""

import numpy as np


class SamplingHandler():

    def __init__(self, game, dist, policy, gamma):
        self.game = game
        self.dist = dist
        self.policy = policy
        self.gamma = gamma

    def acceptState(self):
        """
        Return true if the state is accepted
        With prob. self.gamma return true
        """
        return np.random.random() < self.gamma

    def getStartState(self):
        """
        Return a state
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
        Return a sampled state
        Get a sample state from a rollout with prob. self.gamma
        """
        iteration = 1
        currState = startState
        while iteration <= cutOff:
            currState = self.game.getNextStateWithAction(currState, self.policy[currState])
            if self.acceptState():
                return currState

        return currState

    def sampleStates(self, stateNum, cutOff):
        """
        Return list of states
        Sample stateNum states.
        In each rollout, at most cutOff steps will be excuted.
        """
        sampledStates = []
        for i in range(stateNum):
            startState = self.getStartState()
            sampledState = self.getSampledState(startState, cutOff)
            sampledStates.append(sampledState)

        return sampledStates
